import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib

# Xử lý lỗi load model
@st.cache_resource
def load_model():
    try:
        # Thử joblib trước
        model = joblib.load("house_price_model.pkl")
        return model
    except Exception as e:
        st.error(f"Lỗi khi load model với joblib: {e}")
        try:
            # Thử pickle nếu joblib không work
            with open("house_price_model.pkl", "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e2:
            st.error(f"Lỗi khi load model với pickle: {e2}")
            st.error("Vui lòng kiểm tra lại file model hoặc cài đặt lại dependencies")
            st.stop()

# Load mô hình đã huấn luyện
model = load_model()

st.title("🏡 Dự đoán giá nhà - House Price Predictor")
st.markdown("Nhập các thông số bên dưới để dự đoán giá:")

# Tạo form nhập liệu với columns để layout đẹp hơn
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Chất lượng tổng thể (OverallQual)", 1, 10, 5)
    gr_liv_area = st.number_input("Diện tích sàn (GrLivArea)", value=1500, min_value=0)
    total_bsmt_sf = st.number_input("Tổng Basement (TotalBsmtSF)", value=800, min_value=0)
    bsmt_fin_sf1 = st.number_input("Basement hoàn thiện (BsmtFinSF1)", value=500, min_value=0)
    garage_cars = st.slider("Số xe trong garage (GarageCars)", 0, 5, 2)
    garage_area = st.number_input("Diện tích garage (GarageArea)", value=400, min_value=0)

with col2:
    first_flr_sf = st.number_input("1st Floor SF", value=1000, min_value=0)
    second_flr_sf = st.number_input("2nd Floor SF", value=500, min_value=0)
    lot_area = st.number_input("Diện tích đất (LotArea)", value=8000, min_value=0)
    year_built = st.number_input("Năm xây dựng (YearBuilt)", min_value=1900, max_value=2025, value=2000)
    year_remod_add = st.number_input("Năm cải tạo (YearRemodAdd)", min_value=1900, max_value=2025, value=2005)
    tot_rms_abv_grd = st.slider("Tổng số phòng trên mặt đất (TotRmsAbvGrd)", 2, 15, 6)

# Validation input
if year_remod_add < year_built:
    st.warning("⚠️ Năm cải tạo không thể nhỏ hơn năm xây dựng!")
    year_remod_add = year_built

if bsmt_fin_sf1 > total_bsmt_sf:
    st.warning("⚠️ Basement hoàn thiện không thể lớn hơn tổng basement!")
    bsmt_fin_sf1 = total_bsmt_sf

if garage_area > 0 and garage_cars == 0:
    st.info("💡 Có diện tích garage nhưng không có xe - tự động đặt 1 xe")
    garage_cars = 1

# Tạo DataFrame input
current_year = 2025
input_data = {
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'TotalBsmtSF': total_bsmt_sf,
    'BsmtFinSF1': bsmt_fin_sf1,
    'GarageCars': garage_cars,
    'GarageArea': garage_area,
    '1stFlrSF': first_flr_sf,
    '2ndFlrSF': second_flr_sf,
    'LotArea': lot_area,
    'YearBuilt': year_built,
    'YearRemodAdd': year_remod_add,
    'TotRmsAbvGrd': tot_rms_abv_grd,
    
    # Các feature được tạo thêm
    'ChatLuong_x_DienTich': overall_qual * gr_liv_area,
    'TyLe_Basement': total_bsmt_sf / (gr_liv_area + 1),
    'TyLe_Basement_HoanThien': bsmt_fin_sf1 / (total_bsmt_sf + 1),
    'Tuoi_Nha': current_year - year_built,
    'Tuoi_CaiTao': current_year - year_remod_add,
    'Thoi_Gian_CaiTao': year_remod_add - year_built,
    'DienTich_MoiPhong': gr_liv_area / (tot_rms_abv_grd + 1),
    'DienTich_Garage_MoiXe': garage_area / (garage_cars + 1),
    'Da_CaiTao': int(year_remod_add > year_built),
    'Co_Basement': int(total_bsmt_sf > 0),
    'Co_Basement_HoanThien': int(bsmt_fin_sf1 > 0),
    'Tong_DienTich_Tang': first_flr_sf + second_flr_sf,
}

# Tạo DataFrame
input_df = pd.DataFrame([input_data])

# Hiển thị preview data
with st.expander("🔍 Xem dữ liệu đầu vào"):
    st.dataframe(input_df)

# Dự đoán khi nhấn nút
if st.button("🚀 Dự đoán giá nhà", type="primary"):
    try:
        with st.spinner("Đang tính toán..."):
            prediction = model.predict(input_df)[0]
            
        # Hiển thị kết quả
        st.success(f"💰 **Giá nhà dự đoán: ${prediction:,.0f}**")
        
        # Thêm một số thông tin bổ sung
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Giá/m²", f"${prediction/gr_liv_area:,.0f}")
        with col2:
            st.metric("Tuổi nhà", f"{current_year - year_built} năm")
        with col3:
            st.metric("Chất lượng", f"{overall_qual}/10")
            
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        st.info("Vui lòng kiểm tra lại dữ liệu đầu vào")

# Footer
st.markdown("---")
st.markdown("*Được phát triển bằng Streamlit và Machine Learning*")