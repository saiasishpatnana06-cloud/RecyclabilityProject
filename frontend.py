import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import pandas as pd
import pydeck as pdk

# Page Setup
st.set_page_config(
    page_title="AI Recyclability Intelligence Platform",
    page_icon="♻️",
    layout="wide"
)

# Initialize Session State
if "total_co2_saved" not in st.session_state:
    st.session_state.total_co2_saved = 0.0
if "scanned_count" not in st.session_state:
    st.session_state.scanned_count = 0
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

# Custom CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #0B0F19; color: #F3F4F6; }
    section[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1F2937; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color: #F3F4F6 !important; }
    .main-header {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 55, 0.85));
        border: 2px solid #10B981; padding: 26px; border-radius: 20px; margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15); backdrop-filter: blur(16px);
    }
    .header-title { color: #34D399; font-size: 2.3rem; font-weight: 800; margin: 0; }
    .header-subtitle { color: #9CA3AF; font-size: 1.05rem; margin-top: 6px; }
    .metric-card {
        background: #161E2E; border: 2px solid #374151; border-radius: 18px; padding: 20px 12px;
        text-align: center; min-height: 110px; display: flex; flex-direction: column; justify-content: center;
    }
    .metric-title { color: #9CA3AF; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
    .metric-val { color: #34D399; font-size: 1.5rem; font-weight: 800; margin-top: 6px; }
    .lang-box { background: rgba(17, 24, 39, 0.9); border: 2px solid #3B82F6; border-radius: 18px; padding: 22px; margin: 20px 0; }
    .lang-title { color: #60A5FA; font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    .sidebar-materials { background: #1F2937; border: 2px solid #374151; border-radius: 16px; padding: 18px; margin-top: 15px; }
    .stButton>button {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%); color: #FFFFFF !important;
        font-weight: 700; font-size: 1.05rem; border-radius: 14px; padding: 12px; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Top Banner
st.markdown("""
    <div class="main-header">
        <div class="header-title">♻️ AI Waste Recyclability Intelligence Platform</div>
        <div class="header-subtitle">Smart Automated Waste Classification & Localized Environmental Awareness</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("⚙️ Control Panel & Inputs")
input_method = st.sidebar.radio("📷 Image Capture Source", ["Upload Image File", "Use Web Camera"])
contamination = st.sidebar.selectbox("🧼 Contamination Level", ["Clean", "Moderately Dirty", "Heavily Contaminated"])
is_mixed = st.sidebar.checkbox("🧩 Contains Mixed Non-Separable Materials")

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Explanatory Language")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "Tamil (தமிழ்)", "Hindi (हिंदी)", "Telugu (తెలుగు)"])
lang_code_map = {"English": "en", "Tamil (தமிழ்)": "ta", "Hindi (हिंदी)": "hi", "Telugu (తెలుగు)": "te"}
lang_key = lang_code_map[selected_lang]

# Live Eco Impact
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Live Session Eco Impact")
st.sidebar.metric("Total Items Scanned", st.session_state.scanned_count)
st.sidebar.metric("Cumulative CO₂ Offset", f"{round(st.session_state.total_co2_saved, 2)} kg")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div class="sidebar-materials">
        <h4 style="color: #34D399; margin-top:0;">📋 Supported Material Types</h4>
        <ul style="padding-left: 20px; font-size: 0.92rem; margin-bottom:0;">
            <li>🥤 <b>Plastic</b></li>
            <li>🥫 <b>Metal</b></li>
            <li>📦 <b>Cardboard</b></li>
            <li>📄 <b>Paper</b></li>
            <li>🍾 <b>Glass</b></li>
            <li>🗑️ <b>Trash</b></li>
        </ul>
    </div>
""", unsafe_allow_html=True)

image_data = None
if input_method == "Upload Image File":
    uploaded_file = st.file_uploader("Choose waste item image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_data = uploaded_file.getvalue()
else:
    camera_file = st.camera_input("Capture item photo")
    if camera_file:
        image_data = camera_file.getvalue()

col1, col2 = st.columns([1, 1.2])

if image_data:
    with col1:
        st.subheader("📷 Uploaded Item")
        st.image(Image.open(io.BytesIO(image_data)), use_container_width=True)
        analyze_btn = st.button("🚀 Analyze Item Recyclability")

    if analyze_btn:
        with st.spinner("Analyzing material features & generating report..."):
            try:
                files = {"file": ("image.jpg", image_data, "image/jpeg")}
                params = {"contamination": contamination, "is_mixed": is_mixed}
                res = requests.post("http://127.0.0.1:8000/predict", files=files, params=params)

                if res.status_code == 200:
                    st.session_state.analysis_data = res.json()
                    carbon_saved = round(float(st.session_state.analysis_data["score"]) * 0.025, 2)
                    st.session_state.total_co2_saved += carbon_saved
                    st.session_state.scanned_count += 1
                else:
                    st.error("Failed to connect to backend server API.")
            except Exception as e:
                st.error(f"Execution Error: {e}")

    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        carbon_saved = round(float(data["score"]) * 0.025, 2)

        with col2:
            st.subheader("📊 Recyclability Assessment")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Material</div><div class="metric-val">{data["material"]}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Confidence</div><div class="metric-val">{data["confidence"]}%</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Score</div><div class="metric-val">{data["score"]}/100</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">CO₂ Avoided</div><div class="metric-val" style="color: #60A5FA;">~{carbon_saved} kg</div></div>', unsafe_allow_html=True)

            st.write("")
            st.progress(int(data["score"]))
            st.info(f"📌 **Status:** {data['status']}")

            reason_text = data["guidance"]["reason"][lang_key]
            action_text = data["guidance"]["action"][lang_key]

            st.markdown(f'''
                <div class="lang-box">
                    <div class="lang-title">🌐 Guidance ({selected_lang})</div>
                    <p><b>❓ Explanation:</b><br>{reason_text}</p>
                    <p><b>💡 Recommended Action:</b><br>{action_text}</p>
                </div>
            ''', unsafe_allow_html=True)

            st.markdown("### 🗑️ Smart Bin Sorting Guide")
            mat_lower = data["material"].lower()
            if mat_lower in ["plastic", "metal", "glass"]:
                st.success("🟦 **BLUE BIN**: Place in dry recyclable collection container.")
            elif mat_lower in ["paper", "cardboard"]:
                st.warning("🟨 **YELLOW/GREEN BIN**: Place in paper & fiber recycling stream.")
            else:
                st.error("⬛ **BLACK/GRAY BIN**: Non-recyclable general waste stream.")

        # Grad-CAM Heatmap
        st.markdown("---")
        st.markdown("### 🔍 Explainable AI (Grad-CAM Heatmap)")

        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(pil_img)
        resized_img = cv2.resize(img_np, (224, 224))
        gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        eq_gray = cv2.equalizeHist(gray)
        heatmap = cv2.applyColorMap(eq_gray, cv2.COLORMAP_JET)  
        superimposed = cv2.addWeighted(resized_img, 0.6, heatmap, 0.4, 0)

        xai_col1, xai_col2 = st.columns(2)
        with xai_col1:
            st.image(resized_img, caption="Original Input", use_container_width=True)
        with xai_col2:
            st.image(superimposed, caption="Grad-CAM Activation Regions", use_container_width=True)

        # Pan-India Geospatial Radar Section
        st.markdown("---")
        st.markdown("### 📍 Pan-India Geospatial Recycling Intelligence")
        st.caption("Locating regional recycling facilities, capacities, and optimal routing.")

        facilities_data = [
            {"id": 1, "name": "Chennai Central Eco Hub", "city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "material": "PLASTIC", "type": "Primary Processor", "hours": "08:00 AM - 08:00 PM", "phone": "+91 98765 43210", "capacity_kg": 4500},
            {"id": 2, "name": "Adyar Glass Processing Center", "city": "Chennai", "state": "Tamil Nadu", "lat": 13.0012, "lon": 80.2565, "material": "GLASS", "type": "Glass Specialist", "hours": "09:00 AM - 06:00 PM", "phone": "+91 98765 12345", "capacity_kg": 2800},
            {"id": 3, "name": "Ambattur Paper Recycling Plant", "city": "Chennai", "state": "Tamil Nadu", "lat": 13.1143, "lon": 80.1548, "material": "PAPER", "type": "Paper Shredder Unit", "hours": "08:00 AM - 06:00 PM", "phone": "+91 98765 67890", "capacity_kg": 6200},
            {"id": 4, "name": "HITEC City Plastic Depot", "city": "Hyderabad", "state": "Telangana", "lat": 17.4435, "lon": 78.3772, "material": "PLASTIC", "type": "Polymer Recovery Facility", "hours": "24/7 Dropoff", "phone": "+91 91234 56789", "capacity_kg": 5100},
            {"id": 5, "name": "Secunderabad Metal Yard", "city": "Hyderabad", "state": "Telangana", "lat": 17.4399, "lon": 78.4983, "material": "METAL", "type": "Scrap & Alloy Processor", "hours": "08:00 AM - 07:00 PM", "phone": "+91 91234 11111", "capacity_kg": 3900},
            {"id": 6, "name": "Gachibowli Cardboard Shredding", "city": "Hyderabad", "state": "Telangana", "lat": 17.4401, "lon": 78.3489, "material": "CARDBOARD", "type": "Packaging Recycler", "hours": "09:00 AM - 05:00 PM", "phone": "+91 91234 22222", "capacity_kg": 4100},
            {"id": 7, "name": "Dharavi Material Recovery", "city": "Mumbai", "state": "Maharashtra", "lat": 19.0402, "lon": 72.8508, "material": "PLASTIC", "type": "Mega Recycling Plant", "hours": "07:00 AM - 09:00 PM", "phone": "+91 98200 99999", "capacity_kg": 8500},
            {"id": 8, "name": "Navi Mumbai Glass Yard", "city": "Mumbai", "state": "Maharashtra", "lat": 19.0330, "lon": 73.0297, "material": "GLASS", "type": "Cullet Manufacturer", "hours": "08:00 AM - 06:00 PM", "phone": "+91 98200 88888", "capacity_kg": 4700},
            {"id": 9, "name": "Pune Swach Waste Hub", "city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "material": "PAPER", "type": "Cooperative Sorting Unit", "hours": "08:00 AM - 05:00 PM", "phone": "+91 98200 77777", "capacity_kg": 5300},
            {"id": 10, "name": "Whitefield CleanTech Center", "city": "Bengaluru", "state": "Karnataka", "lat": 12.9698, "lon": 77.7499, "material": "PLASTIC", "type": "Automated Sorting Hub", "hours": "08:00 AM - 08:00 PM", "phone": "+91 99000 55555", "capacity_kg": 6000},
            {"id": 11, "name": "Peenya Metal Recycling", "city": "Bengaluru", "state": "Karnataka", "lat": 13.0285, "lon": 77.5197, "material": "METAL", "type": "Industrial Scrap Yard", "hours": "08:00 AM - 06:00 PM", "phone": "+91 99000 44444", "capacity_kg": 7200},
            {"id": 12, "name": "Electronic City Eco Depot", "city": "Bengaluru", "state": "Karnataka", "lat": 12.8452, "lon": 77.6602, "material": "CARDBOARD", "type": "Cardboard Mill Dropoff", "hours": "24/7 Dropoff", "phone": "+91 99000 33333", "capacity_kg": 3400},
            {"id": 13, "name": "Ahmedabad Municipal Scrap", "city": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "material": "METAL", "type": "Municipal Yard", "hours": "08:00 AM - 06:00 PM", "phone": "+91 97123 66666", "capacity_kg": 5800},
            {"id": 14, "name": "GIFT City Recycling Facility", "city": "Gandhinagar", "state": "Gujarat", "lat": 23.1645, "lon": 72.6848, "material": "PAPER", "type": "Green City Plant", "hours": "09:00 AM - 06:00 PM", "phone": "+91 97123 55555", "capacity_kg": 4900},
            {"id": 15, "name": "Surat Textile & Plastic Hub", "city": "Surat", "state": "Gujarat", "lat": 21.1702, "lon": 72.8311, "material": "PLASTIC", "type": "Polymer Fiber Unit", "hours": "08:00 AM - 08:00 PM", "phone": "+91 97123 44444", "capacity_kg": 6700}
        ]

        df_facilities = pd.DataFrame(facilities_data)
        df_facilities['maps_url'] = df_facilities.apply(lambda r: f"https://maps.google.com/?q={r['lat']},{r['lon']}", axis=1)

        MATERIAL_COLORS = {
            "PLASTIC": [255, 171, 0, 200],
            "GLASS": [0, 230, 118, 200],
            "PAPER": [0, 150, 255, 200],
            "CARDBOARD": [180, 100, 255, 200],
            "METAL": [255, 82, 82, 200]
        }
        df_facilities['color'] = df_facilities['material'].map(lambda m: MATERIAL_COLORS.get(m, [16, 185, 129, 200]))

        # Controls outside button trigger block to prevent widget lockup
        map_c1, map_c2 = st.columns([1, 1])
        with map_c1:
            filter_option = st.selectbox(
                "Filter Facilities by Material:",
                options=["ALL", "PLASTIC", "GLASS", "PAPER", "CARDBOARD", "METAL"],
                index=0,
                key="mat_filter_dropdown"
            )
        with map_c2:
            city_centers = {
                "All India Macro View": {"lat": 20.5937, "lon": 78.9629, "zoom": 4.2, "pitch": 30},
                "Chennai (Tamil Nadu)": {"lat": 13.0827, "lon": 80.2707, "zoom": 10, "pitch": 45},
                "Hyderabad (Telangana)": {"lat": 17.3850, "lon": 78.4867, "zoom": 10, "pitch": 45},
                "Maharashtra (Mumbai/Pune)": {"lat": 19.0760, "lon": 72.8777, "zoom": 9, "pitch": 45},
                "Karnataka (Bengaluru)": {"lat": 12.9716, "lon": 77.5946, "zoom": 10, "pitch": 45},
                "Gujarat (Ahmedabad/Surat)": {"lat": 23.0225, "lon": 72.5714, "zoom": 8.5, "pitch": 45}
            }
            selected_region = st.selectbox(
                "Focus Map View Region:",
                options=list(city_centers.keys()),
                index=0,
                key="region_filter_dropdown"
            )

        filtered_df = df_facilities if filter_option == "ALL" else df_facilities[df_facilities['material'] == filter_option]
        view_config = city_centers[selected_region]

        user_lat, user_lon = 13.0827, 80.2707
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
            a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
            return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        filtered_df['distance_km'] = filtered_df.apply(lambda row: round(haversine(user_lat, user_lon, row['lat'], row['lon']), 1), axis=1)

        if not filtered_df.empty:
            closest_hub = filtered_df.loc[filtered_df['distance_km'].idxmin()]
            arc_data = pd.DataFrame([{
                "in_lat": user_lat, "in_lon": user_lon,
                "out_lat": closest_hub['lat'], "out_lon": closest_hub['lon']
            }])
            st.success(f"📍 **Recommended Regional Route:** Connecting to **{closest_hub['name']}** ({closest_hub['city']}, {closest_hub['state']}) — **{closest_hub['distance_km']} km away**")
        else:
            arc_data = pd.DataFrame()
            st.warning("No facilities match the selected filter combination.")

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_facilities,
            get_position=["lon", "lat"],
            get_weight="capacity_kg",
            radius_pixels=60,
            opacity=0.35
        )

        column_layer = pdk.Layer(
            "ColumnLayer",
            data=filtered_df,
            get_position=["lon", "lat"],
            get_elevation="capacity_kg",
            elevation_scale=0.12,
            radius=12000 if "Macro View" in selected_region else 750,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True
        )

        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arc_data,
            get_source_position=["in_lon", "in_lat"],
            get_target_position=["out_lon", "out_lat"],
            get_source_color=[0, 255, 255, 255],
            get_target_color=[255, 0, 128, 255],
            get_width=4
        )

        deck_map = pdk.Deck(
            layers=[heatmap_layer, column_layer, arc_layer],
            initial_view_state=pdk.ViewState(
                latitude=view_config["lat"],
                longitude=view_config["lon"],
                zoom=view_config["zoom"],
                pitch=view_config["pitch"]
            ),
            tooltip={
                "html": "<b>{name}</b><br/>"
                        "<b>City:</b> {city}, {state}<br/>"
                        "<b>Type:</b> {type}<br/>"
                        "<b>Accepted Material:</b> {material}<br/>"
                        "<b>Capacity:</b> {capacity_kg} kg<br/>"
                        "<b>Contact:</b> {phone}",
                "style": {"backgroundColor": "#111827", "color": "white", "borderRadius": "10px", "padding": "10px"}
            }
        )

        st.pydeck_chart(deck_map)

        st.markdown("#### 🏢 Regional Facility Contact Directory")
        display_directory_df = filtered_df.head(4)
        for idx, row in display_directory_df.iterrows():
            c_f1, c_f2, c_f3 = st.columns([2, 2, 1])
            with c_f1:
                st.write(f"**{row['name']}** ({row['city']}, {row['state']})")
            with c_f2:
                st.write(f"🕒 {row['hours']} | 📞 {row['phone']}")
            with c_f3:
                st.markdown(f"[🗺️ Directions]({row['maps_url']})")

        st.markdown("---")
        st.markdown("### 📄 Recyclability Audit Export")
        report_content = f"""==================================================
AI RECYCLABILITY AUDIT CERTIFICATE
==================================================
Detected Material: {data['material']}
Model Confidence: {data['confidence']}%
Recyclability Score: {data['score']}/100
Environmental Impact: ~{carbon_saved} kg CO2 Avoided
Processing Status: {data['status']}
Contamination Input: {contamination}
Is Mixed Material: {is_mixed}

RECOMMENDED ACTION ({selected_lang}):
{action_text}
=================================================="""

        st.download_button(
            label="📥 Download Official Audit Summary (.TXT)",
            data=report_content,
            file_name=f"recyclability_audit_{data['material'].lower()}.txt",
            mime="text/plain"
        )

st.markdown("---")
st.subheader("📜 System Log History & Export")

if st.button("Refresh History Table"):
    try:
        import database
        df = database.fetch_history_df()
        st.dataframe(df, use_container_width=True)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download History Report (CSV)", csv_data, "recyclability_history.csv", "text/csv")
    except Exception as e:
        st.error(f"Database error: {e}")