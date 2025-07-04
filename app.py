import streamlit as st
import numpy as np
import joblib

# Load mô hình đã huấn luyện
model = joblib.load("house_price_model.pkl")

st.title("🏡 Dự đoán giá nhà - House Price Predictor")

st.markdown("Nhập các thông số bên dưới để dự đoán giá:")

# Tạo form nhập liệu
overall_qual = st.slider("Chất lượng tổng thể (OverallQual)", 1, 10, 5)
gr_liv_area = st.number_input("Diện tích sàn (GrLivArea)", value=1500)
total_bsmt_sf = st.number_input("Tổng Basement (TotalBsmtSF)", value=800)
bsmt_fin_sf1 = st.number_input("Basement hoàn thiện (BsmtFinSF1)", value=500)
garage_cars = st.slider("Số xe trong garage (GarageCars)", 0, 5, 2)
garage_area = st.number_input("Diện tích garage (GarageArea)", value=400)
first_flr_sf = st.number_input("1st Floor SF", value=1000)
second_flr_sf = st.number_input("2nd Floor SF", value=500)
lot_area = st.number_input("Diện tích đất (LotArea)", value=8000)
year_built = st.number_input("Năm xây dựng (YearBuilt)", min_value=1900, max_value=2025, value=2000)
year_remod_add = st.number_input("Năm cải tạo (YearRemodAdd)", min_value=1900, max_value=2025, value=2005)
tot_rms_abv_grd = st.slider("Tổng số phòng trên mặt đất (TotRmsAbvGrd)", 2, 15, 6)

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

import pandas as pd
input_df = pd.DataFrame([input_data])

# Dự đoán khi nhấn nút
if st.button("Dự đoán giá nhà"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Giá nhà dự đoán: ${prediction:,.0f}")
