import pandas as pd
import numpy as np
import io
import joblib
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.impute import SimpleImputer

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API để dự đoán giá nhà sử dụng Polynomial Features + ElasticNet",
    version="1.0.0"
)

# --- LOAD MODEL VÀ LẤY DANH SÁCH FEATURE TỪ MODEL ---
try:
    model = joblib.load('house_price_model.pkl')
    print("✅ Model loaded successfully!")
    
    # Lấy danh sách tên feature mà model đã được huấn luyện
    if hasattr(model, 'feature_names_in_'):
        MODEL_EXPECTED_FEATURES = list(model.feature_names_in_)
        print(f"✅ Model expects {len(MODEL_EXPECTED_FEATURES)} features")
        print(f"📋 First 10 features: {MODEL_EXPECTED_FEATURES[:10]}")
    else:
        # Fallback: Định nghĩa manual nếu cần
        MODEL_EXPECTED_FEATURES = None
        print("❌ Warning: Model does not have 'feature_names_in_'")

except FileNotFoundError:
    print("❌ Model file not found! Please train and save model first.")
    model = None
    MODEL_EXPECTED_FEATURES = None

# --- PYDANTIC MODELS ---
class HouseFeatures(BaseModel):
    """Model đầu vào cho prediction đơn lẻ"""
    OverallQual: int
    GrLivArea: int
    TotalBsmtSF: int
    BsmtFinSF1: int
    GarageCars: int
    GarageArea: int
    FirstFlrSF: int  # Sẽ được convert thành 1stFlrSF
    SecondFlrSF: int  # Sẽ được convert thành 2ndFlrSF
    LotArea: int
    YearBuilt: int
    YearRemodAdd: int
    TotRmsAbvGrd: int
    
    class Config:
        schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "TotalBsmtSF": 900,
                "BsmtFinSF1": 700,
                "GarageCars": 2,
                "GarageArea": 480,
                "FirstFlrSF": 800,
                "SecondFlrSF": 700,
                "LotArea": 8000,
                "YearBuilt": 2000,
                "YearRemodAdd": 2000,
                "TotRmsAbvGrd": 7
            }
        }

class PredictionResponse(BaseModel):
    """Model response cho prediction"""
    predicted_price: float
    formatted_price: str
    confidence: str
    model_version: str = "Polynomial Features + ElasticNet v1.0"

# --- HELPER FUNCTIONS ---
def create_features_for_prediction(input_data: pd.DataFrame):
    """Tạo các features mới từ dữ liệu đầu vào, giống như trong quá trình training"""
    nam_hien_tai = 2025
    df_moi = input_data.copy()
    
    # 🔍 DEBUG: Kiểm tra NaN values
    print(f"📊 DataFrame shape: {df_moi.shape}")
    print(f"🔍 NaN values found: {df_moi.isnull().sum().sum()}")
    if df_moi.isnull().sum().sum() > 0:
        print("❌ Columns with NaN:")
        nan_cols = df_moi.columns[df_moi.isnull().any()].tolist()
        for col in nan_cols:
            print(f"   - {col}: {df_moi[col].isnull().sum()} NaN values")
    
    # 🛠️ XỬ LÝ MISSING VALUES TRƯỚC KHI TẠO FEATURES
    # Thay thế các giá trị string missing bằng NaN
    df_moi = df_moi.replace(['', 'NULL', 'N/A', 'null', 'None', 'na', 'NA'], np.nan)
    
    # Fill NaN values với giá trị hợp lý
    numeric_columns = df_moi.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_moi[col].isnull().sum() > 0:
            if col in ['TotalBsmtSF', 'BsmtFinSF1', 'GarageArea', 'GarageCars']:
                # Basement và garage có thể không có -> fill 0
                df_moi[col] = df_moi[col].fillna(0)
            elif col in ['YearRemodAdd']:
                # Nếu không có remodel -> dùng YearBuilt
                df_moi[col] = df_moi[col].fillna(df_moi.get('YearBuilt', 2000))
            else:
                # Các cột khác -> dùng median
                df_moi[col] = df_moi[col].fillna(df_moi[col].median())
    
    print(f"✅ After cleaning: {df_moi.isnull().sum().sum()} NaN values")

    # Đổi tên cột để khớp với training
    if 'FirstFlrSF' in df_moi.columns:
        df_moi['1stFlrSF'] = df_moi['FirstFlrSF']
        df_moi = df_moi.drop('FirstFlrSF', axis=1)
    if 'SecondFlrSF' in df_moi.columns:
        df_moi['2ndFlrSF'] = df_moi['SecondFlrSF']
        df_moi = df_moi.drop('SecondFlrSF', axis=1)

    # 1. Tương tác giữa chất lượng và diện tích
    if 'OverallQual' in df_moi.columns and 'GrLivArea' in df_moi.columns:
        df_moi['ChatLuong_x_DienTich'] = df_moi['OverallQual'] * df_moi['GrLivArea']
    
    # 2. Tỷ lệ diện tích basement
    if 'TotalBsmtSF' in df_moi.columns and 'GrLivArea' in df_moi.columns:
        df_moi['TyLe_Basement'] = df_moi['TotalBsmtSF'] / (df_moi['GrLivArea'] + 1)
    
    if 'BsmtFinSF1' in df_moi.columns and 'TotalBsmtSF' in df_moi.columns:
        df_moi['TyLe_Basement_HoanThien'] = df_moi['BsmtFinSF1'] / (df_moi['TotalBsmtSF'] + 1)
    
    # 3. Tuổi nhà và thời gian cải tạo
    if 'YearBuilt' in df_moi.columns:
        df_moi['Tuoi_Nha'] = nam_hien_tai - df_moi['YearBuilt']
    
    if 'YearRemodAdd' in df_moi.columns and 'YearBuilt' in df_moi.columns:
        df_moi['Tuoi_CaiTao'] = nam_hien_tai - df_moi['YearRemodAdd']
        df_moi['Thoi_Gian_CaiTao'] = df_moi['YearRemodAdd'] - df_moi['YearBuilt']
    
    # 4. Diện tích trung bình mỗi phòng
    if 'GrLivArea' in df_moi.columns and 'TotRmsAbvGrd' in df_moi.columns:
        df_moi['DienTich_MoiPhong'] = df_moi['GrLivArea'] / (df_moi['TotRmsAbvGrd'] + 1)
    
    # 5. Chỉ số garage
    if 'GarageArea' in df_moi.columns and 'GarageCars' in df_moi.columns:
        df_moi['DienTich_Garage_MoiXe'] = df_moi['GarageArea'] / (df_moi['GarageCars'] + 1)
    
    # 6. Features phân loại
    if 'YearRemodAdd' in df_moi.columns and 'YearBuilt' in df_moi.columns:
        df_moi['Da_CaiTao'] = (df_moi['YearRemodAdd'] > df_moi['YearBuilt']).astype(int)
    
    if 'TotalBsmtSF' in df_moi.columns:
        df_moi['Co_Basement'] = (df_moi['TotalBsmtSF'] > 0).astype(int)
    
    if 'BsmtFinSF1' in df_moi.columns:
        df_moi['Co_Basement_HoanThien'] = (df_moi['BsmtFinSF1'] > 0).astype(int)
    
    # 7. Features tổng hợp
    if all(col in df_moi.columns for col in ['1stFlrSF', '2ndFlrSF']):
        df_moi['Tong_DienTich_Tang'] = df_moi['1stFlrSF'] + df_moi['2ndFlrSF']
    
    # Chỉ giữ lại các cột là số
    df_numeric = df_moi.select_dtypes(include=[np.number])
    
    return df_numeric

def calculate_confidence(overall_qual: int, gr_liv_area: int, total_bsmt_sf: int = 0) -> str:
    """Tính toán độ tin cậy của prediction"""
    if overall_qual >= 8 and gr_liv_area >= 1500:
        return "High"
    elif overall_qual >= 6 and gr_liv_area >= 1000:
        return "Medium"
    else:
        return "Low"

# --- API ENDPOINTS ---
@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "🏠 House Price Prediction API",
        "version": "1.0.0",
        "status": "active" if model is not None else "model_not_loaded",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch", 
            "upload_csv": "/upload-csv",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "features_available": MODEL_EXPECTED_FEATURES is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_house(house: HouseFeatures):
    """Dự đoán giá cho một ngôi nhà"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if MODEL_EXPECTED_FEATURES is None:
        raise HTTPException(status_code=500, detail="Model features not identified")
    
    try:
        # Chuyển đổi input thành DataFrame
        input_dict = house.dict()
        df = pd.DataFrame([input_dict])
        
        # Tạo features mới
        df_processed = create_features_for_prediction(df)
        
        # Đảm bảo có đủ features cho model
        for col in MODEL_EXPECTED_FEATURES:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Lọc và sắp xếp features theo thứ tự model mong đợi
        X_predict = df_processed[MODEL_EXPECTED_FEATURES]
        
        # Dự đoán
        prediction = model.predict(X_predict)[0]
        
        # Tính confidence
        confidence = calculate_confidence(
            house.OverallQual, 
            house.GrLivArea, 
            house.TotalBsmtSF
        )
        
        return PredictionResponse(
            predicted_price=float(prediction),
            formatted_price=f"${prediction:,.0f}",
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch_houses(houses: List[HouseFeatures]):
    """Dự đoán giá cho nhiều ngôi nhà"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if MODEL_EXPECTED_FEATURES is None:
        raise HTTPException(status_code=500, detail="Model features not identified")
    
    try:
        # Chuyển đổi list thành DataFrame
        input_data = [house.dict() for house in houses]
        df = pd.DataFrame(input_data)
        
        # Tạo features mới
        df_processed = create_features_for_prediction(df)
        
        # Đảm bảo có đủ features cho model
        for col in MODEL_EXPECTED_FEATURES:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Lọc và sắp xếp features theo thứ tự model mong đợi
        X_predict = df_processed[MODEL_EXPECTED_FEATURES]
        
        # Dự đoán
        predictions = model.predict(X_predict)
        
        # Tạo response
        results = []
        for i, (house, prediction) in enumerate(zip(houses, predictions)):
            confidence = calculate_confidence(
                house.OverallQual, 
                house.GrLivArea, 
                house.TotalBsmtSF
            )
            
            results.append({
                "index": i,
                "predicted_price": float(prediction),
                "formatted_price": f"${prediction:,.0f}",
                "confidence": confidence
            })
        
        return {"predictions": results, "total_count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/debug-csv")
async def debug_csv(file: UploadFile = File(...)):
    """Debug endpoint để kiểm tra CSV file"""
    try:
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # Thử detect separator
        if ';' in content_str.split('\n')[0]:
            df = pd.read_csv(io.StringIO(content_str), sep=';')
            separator = "semicolon"
        else:
            df = pd.read_csv(io.StringIO(content_str))
            separator = "comma"
        
        # Thông tin chi tiết về file
        debug_info = {
            "file_info": {
                "filename": file.filename,
                "size_chars": len(content_str),
                "separator": separator,
                "shape": df.shape
            },
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "non_numeric_columns": df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@app.post("/upload-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Upload CSV file và dự đoán giá cho tất cả nhà trong file"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if MODEL_EXPECTED_FEATURES is None:
        raise HTTPException(status_code=500, detail="Model feature names not identified")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Đọc file CSV với nhiều options để xử lý encoding
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # 🔍 DEBUG: Kiểm tra nội dung file
        print(f"📄 File size: {len(content_str)} characters")
        print(f"🔍 First 500 characters:")
        print(content_str[:500])
        
        # Thử detect separator
        if ';' in content_str.split('\n')[0]:
            df = pd.read_csv(io.StringIO(content_str), sep=';')
            print("✅ Using semicolon separator")
        else:
            df = pd.read_csv(io.StringIO(content_str))
            print("✅ Using comma separator")
        
        print(f"📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # 🔍 DEBUG: Kiểm tra data types và missing values
        print(f"🔍 Data types:")
        print(df.dtypes)
        print(f"🔍 Missing values per column:")
        print(df.isnull().sum())
        
        # Kiểm tra các cột cần thiết
        required_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 
                          'GarageCars', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
            )
        
        # Tạo features mới
        print("🔧 Creating features...")
        df_processed = create_features_for_prediction(df)
        
        # 🛠️ FINAL NaN CHECK - Đảm bảo không còn NaN
        if df_processed.isnull().sum().sum() > 0:
            print("⚠️ Still have NaN values, applying final imputation...")
            # Sử dụng SimpleImputer để xử lý NaN còn lại
            imputer = SimpleImputer(strategy='median')
            df_processed_array = imputer.fit_transform(df_processed)
            df_processed = pd.DataFrame(df_processed_array, columns=df_processed.columns)
            print("✅ Final imputation completed")
        
        # Đảm bảo có đủ features cho model
        print("🔧 Ensuring all required features...")
        for col in MODEL_EXPECTED_FEATURES:
            if col not in df_processed.columns:
                df_processed[col] = 0
                print(f"   + Added missing feature: {col}")
        
        # Lọc features theo model
        X_predict = df_processed[MODEL_EXPECTED_FEATURES]
        
        # 🔍 FINAL CHECK
        print(f"📊 Final prediction data shape: {X_predict.shape}")
        print(f"🔍 Final NaN check: {X_predict.isnull().sum().sum()}")
        
        if X_predict.isnull().sum().sum() > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Still have NaN values after processing: {X_predict.isnull().sum().to_dict()}"
            )
        
        # Dự đoán
        predictions = model.predict(X_predict)
        
        # Thêm kết quả vào DataFrame gốc
        df['predicted_price'] = predictions
        df['formatted_price'] = df['predicted_price'].apply(lambda x: f"${x:,.0f}")
        df['confidence'] = df.apply(
            lambda row: calculate_confidence(
                row['OverallQual'], 
                row['GrLivArea'], 
                row.get('TotalBsmtSF', 0)
            ), 
            axis=1
        )
        
        # Tạo output CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing error: {str(e)}")

# --- MAIN ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting House Price Prediction API...")
    print("📋 Available endpoints:")
    print("   - GET  /          : Root endpoint")
    print("   - GET  /health    : Health check")
    print("   - POST /predict   : Single prediction")
    print("   - POST /predict-batch : Batch prediction")
    print("   - POST /upload-csv : CSV upload prediction")
    print("🌐 Access docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)