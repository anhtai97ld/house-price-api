import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Load và chuẩn bị dữ liệu từ file CSV
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Đã load dữ liệu: {df.shape[0]} mẫu, {df.shape[1]} cột")
        selected_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', 'GarageCars', 'GarageArea', '1stFlrSF', '2ndFlrSF', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd', 'SalePrice']
        df = df[selected_columns]
        return df
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {file_path}")
        print("💡 Hãy đảm bảo file CSV nằm trong cùng thư mục với main.py")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {str(e)}")
        return None

def create_features(df):
    """
    Tạo các features mới từ dữ liệu gốc
    """
    print("\n🔧 TẠO FEATURES MỚI:")
    print("-" * 40)
    
    # Khởi tạo tham số
    nam_hien_tai = 2025
    df_moi = df.copy()
    
    # 1. Tương tác giữa chất lượng và diện tích
    if 'OverallQual' in df_moi.columns and 'GrLivArea' in df_moi.columns:
        df_moi['ChatLuong_x_DienTich'] = df_moi['OverallQual'] * df_moi['GrLivArea']
        print("   ✓ Tạo tương tác ChấtLượng × DiệnTích")
    
    # 2. Tỷ lệ diện tích basement
    if 'TotalBsmtSF' in df_moi.columns and 'GrLivArea' in df_moi.columns:
        df_moi['TyLe_Basement'] = df_moi['TotalBsmtSF'] / (df_moi['GrLivArea'] + 1)
        print("   ✓ Tạo tỷ lệ diện tích basement")
    
    if 'BsmtFinSF1' in df_moi.columns and 'TotalBsmtSF' in df_moi.columns:
        df_moi['TyLe_Basement_HoanThien'] = df_moi['BsmtFinSF1'] / (df_moi['TotalBsmtSF'] + 1)
        print("   ✓ Tạo tỷ lệ basement hoàn thiện")
    
    # 3. Tuổi nhà và thời gian cải tạo
    if 'YearBuilt' in df_moi.columns:
        df_moi['Tuoi_Nha'] = nam_hien_tai - df_moi['YearBuilt']
        print("   ✓ Tạo tuổi nhà")
    
    if 'YearRemodAdd' in df_moi.columns:
        df_moi['Tuoi_CaiTao'] = nam_hien_tai - df_moi['YearRemodAdd']
        df_moi['Thoi_Gian_CaiTao'] = df_moi['YearRemodAdd'] - df_moi['YearBuilt']
        print("   ✓ Tạo thời gian cải tạo")
    
    # 4. Diện tích trung bình mỗi phòng
    if 'GrLivArea' in df_moi.columns and 'TotRmsAbvGrd' in df_moi.columns:
        df_moi['DienTich_MoiPhong'] = df_moi['GrLivArea'] / (df_moi['TotRmsAbvGrd'] + 1)
        print("   ✓ Tạo diện tích trung bình mỗi phòng")
    
    # 5. Chỉ số garage
    if 'GarageArea' in df_moi.columns and 'GarageCars' in df_moi.columns:
        df_moi['DienTich_Garage_MoiXe'] = df_moi['GarageArea'] / (df_moi['GarageCars'] + 1)
        print("   ✓ Tạo diện tích garage mỗi xe")
    
    # 6. Features phân loại
    if 'YearRemodAdd' in df_moi.columns and 'YearBuilt' in df_moi.columns:
        df_moi['Da_CaiTao'] = (df_moi['YearRemodAdd'] > df_moi['YearBuilt']).astype(int)
        print("   ✓ Tạo indicator đã cải tạo")
    
    if 'TotalBsmtSF' in df_moi.columns:
        df_moi['Co_Basement'] = (df_moi['TotalBsmtSF'] > 0).astype(int)
        print("   ✓ Tạo indicator có basement")
    
    if 'BsmtFinSF1' in df_moi.columns:
        df_moi['Co_Basement_HoanThien'] = (df_moi['BsmtFinSF1'] > 0).astype(int)
        print("   ✓ Tạo indicator basement hoàn thiện")
    
    # 7. Features tổng hợp
    if all(col in df_moi.columns for col in ['1stFlrSF', '2ndFlrSF']):
        df_moi['Tong_DienTich_Tang'] = df_moi['1stFlrSF'] + df_moi['2ndFlrSF']
        print("   ✓ Tạo tổng diện tích các tầng")
    
    print(f"   📊 Tăng từ {df.shape[1]} → {df_moi.shape[1]} features")
    
    return df_moi

def prepare_data(df, target_column='SalePrice', random_state=42):
    """
    Chuẩn bị dữ liệu cho training
    """
    print(f"\n📊 CHUẨN BỊ DỮ LIỆU:")
    print("-" * 40)
    
    # Kiểm tra target column
    if target_column not in df.columns:
        print(f"❌ Không tìm thấy cột target: {target_column}")
        print(f"💡 Các cột có sẵn: {list(df.columns)}")
        return None, None, None, None, None, None, None, None
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Chỉ giữ các cột số
    X_numeric = X.select_dtypes(include=[np.number])
    print(f"📊 Dữ liệu sau xử lý: {X_numeric.shape[0]} mẫu, {X_numeric.shape[1]} features")
    
    # Train-val-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state  # 0.25 * 0.8 = 0.2 của tổng
    )
    
    print(f"📊 Chia dữ liệu:")
    print(f"   ├── Train: {X_train.shape[0]} mẫu ({X_train.shape[0]/len(X_numeric)*100:.1f}%)")
    print(f"   ├── Val:   {X_val.shape[0]} mẫu ({X_val.shape[0]/len(X_numeric)*100:.1f}%)")
    print(f"   └── Test:  {X_test.shape[0]} mẫu ({X_test.shape[0]/len(X_numeric)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_numeric, y

def train_polynomial_model(X_train, y_train, random_state=42):
    """
    Training Polynomial Features model
    """
    print(f"\n🔢 POLYNOMIAL FEATURES MODEL:")
    print("-" * 50)
    
    # Tạo polynomial pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ('scaler', StandardScaler()),
        ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000))
    ])
    
    # Cross-validation (10-fold)
    cv_folds = 10
    poly_scores = cross_val_score(
        poly_pipeline, X_train, y_train, cv=cv_folds,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    poly_rmse = np.sqrt(-poly_scores)
    
    print(f"   📊 Polynomial CV RMSE: {poly_rmse.mean():.4f} ± {poly_rmse.std():.4f}")
    
    # Fit model
    poly_pipeline.fit(X_train, y_train)
    # Lưu model
    import joblib
    joblib.dump(poly_pipeline, 'house_price_model.pkl')
    print("✅ Model đã được lưu!")
    return poly_pipeline, poly_rmse

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Đánh giá model trên tất cả các tập dữ liệu
    """
    print(f"\n📊 ĐÁNH GIÁ CHI TIẾT - POLYNOMIAL MODEL:")
    print("-" * 50)
    
    # Predictions cho cả 3 tập
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics cho training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Metrics cho validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    # Metrics cho test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Tính overfitting gaps
    train_val_r2_gap = train_r2 - val_r2
    train_val_rmse_gap = val_rmse - train_rmse
    val_test_r2_gap = val_r2 - test_r2
    val_test_rmse_gap = test_rmse - val_rmse
    
    print(f"   📈 TRAIN SET:")
    print(f"   ├── RMSE: {train_rmse:.4f}")
    print(f"   ├── R²:   {train_r2:.4f}")
    print(f"   └── MAE:  {train_mae:.4f}")
    
    print(f"   📊 VALIDATION SET:")
    print(f"   ├── RMSE: {val_rmse:.4f}")
    print(f"   ├── R²:   {val_r2:.4f}")
    print(f"   └── MAE:  {val_mae:.4f}")
    
    print(f"   📉 TEST SET:")
    print(f"   ├── RMSE: {test_rmse:.4f}")
    print(f"   ├── R²:   {test_r2:.4f}")
    print(f"   └── MAE:  {test_mae:.4f}")
    
    print(f"   ⚖️ OVERFITTING ANALYSIS:")
    print(f"   ├── Train→Val R² Gap:  {train_val_r2_gap:.4f} {'✅' if train_val_r2_gap < 0.05 else '⚠️' if train_val_r2_gap < 0.1 else '❌'}")
    print(f"   ├── Train→Val RMSE Gap: {train_val_rmse_gap:.4f}")
    print(f"   ├── Val→Test R² Gap:    {val_test_r2_gap:.4f} {'✅' if abs(val_test_r2_gap) < 0.03 else '⚠️'}")
    print(f"   └── Val→Test RMSE Gap:  {val_test_rmse_gap:.4f}")
    
    return {
        'train_rmse': train_rmse, 'train_r2': train_r2, 'train_mae': train_mae,
        'val_rmse': val_rmse, 'val_r2': val_r2, 'val_mae': val_mae,
        'test_rmse': test_rmse, 'test_r2': test_r2, 'test_mae': test_mae,
        'train_val_r2_gap': train_val_r2_gap, 'val_test_r2_gap': val_test_r2_gap
    }

def print_final_summary(cv_rmse, metrics):
    """
    In tóm tắt kết quả cuối cùng
    """
    print(f"\n🎯 TÓM TẮT KẾT QUẢ CUỐI CÙNG:")
    print("=" * 60)
    print(f"✅ Model: Polynomial Features + ElasticNet")
    print(f"✅ CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"✅ Val R²: {metrics['val_r2']:.4f}")
    print(f"✅ Val RMSE: {metrics['val_rmse']:.4f}")
    print(f"✅ Test R²: {metrics['test_r2']:.4f}")
    print(f"✅ Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"✅ Train→Val Gap: {metrics['train_val_r2_gap']:.4f}")
    print(f"✅ Val→Test Gap: {metrics['val_test_r2_gap']:.4f}")

def main():
    """
    Hàm main chạy toàn bộ pipeline
    """
    print("🏠 HOUSE PRICE PREDICTION WITH POLYNOMIAL FEATURES")
    print("=" * 60)
    
    # 1. Load dữ liệu
    # Thay 'your_data.csv' bằng tên file dữ liệu của bạn
    file_path = '/Users/anhtai/house-price-api/train.csv'  # Hoặc 'house_prices.csv', 'data.csv', etc.
    df = load_and_prepare_data(file_path)
    
    if df is None:
        print("❌ Không thể tiếp tục do lỗi load dữ liệu")
        return
    
    # 2. Tạo features mới
    df_with_features = create_features(df)
    
    # 3. Chuẩn bị dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, X_numeric, y = prepare_data(
        df_with_features, target_column='SalePrice', random_state=42
    )
    
    if X_train is None:
        print("❌ Không thể tiếp tục do lỗi chuẩn bị dữ liệu")
        return
    
    # 4. Training model
    model, cv_rmse = train_polynomial_model(X_train, y_train, random_state=42)
    
    # 5. Đánh giá model
    metrics = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 6. Tóm tắt kết quả
    print_final_summary(cv_rmse, metrics)
    
    print(f"\n🎉 HOÀN THÀNH! Model đã được training và đánh giá.")
    print(f"💾 Bạn có thể lưu model bằng: joblib.dump(model, 'house_price_model.pkl')")

if __name__ == "__main__":
    main()


import streamlit as st
import requests # Để gửi yêu cầu HTTP đến API của bạn
import json # Để xử lý dữ liệu JSON

st.title("Ứng Dụng Dự Đoán Giá Nhà")
st.write("Nhập thông tin nhà để nhận dự đoán giá.")

# Giả sử API của bạn đang chạy cục bộ trên cổng 8000
# Khi triển khai, bạn sẽ cần thay đổi URL này thành URL của API đã triển khai
API_URL = "http://127.0.0.1:8000/predict" # Thay đổi nếu API của bạn chạy trên cổng khác hoặc URL khác

# Các trường nhập liệu cho mô hình của bạn (cần khớp với đầu vào API của bạn)
# Ví dụ: Nếu API của bạn mong đợi các trường như 'square_footage', 'num_bedrooms', 'location'
square_footage = st.number_input("Diện tích (sq ft)", min_value=500, max_value=10000, value=1500)
num_bedrooms = st.slider("Số phòng ngủ", 1, 6, 3)
location = st.selectbox("Vị trí", ["Downtown", "Suburb", "Rural"])
# Thêm các trường nhập liệu khác mà mô hình của bạn cần

if st.button("Dự đoán Giá"):
    # Chuẩn bị dữ liệu để gửi đến API
    # Đảm bảo các khóa khớp với tên trường mà API của bạn mong đợi
    data = {
        "square_footage": square_footage,
        "num_bedrooms": num_bedrooms,
        "location": location
        # Thêm các trường dữ liệu khác
    }

    try:
        # Gửi yêu cầu POST đến API của bạn
        response = requests.post(API_URL, json=data)
        response.raise_for_status() # Nâng ngoại lệ cho các mã trạng thái lỗi (4xx hoặc 5xx)

        prediction_result = response.json()
        
        if "predicted_price" in prediction_result:
            st.success(f"Giá nhà dự đoán là: ${prediction_result['predicted_price']:,.2f}")
        else:
            st.error(f"API trả về lỗi hoặc định dạng không mong muốn: {prediction_result}")

    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến API. Hãy đảm bảo API đang chạy.")
    except requests.exceptions.RequestException as e:
        st.error(f"Đã xảy ra lỗi khi gọi API: {e}")
    except json.JSONDecodeError:
        st.error("API trả về phản hồi không phải JSON.")