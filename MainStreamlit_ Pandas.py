import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
import time

# Konfigurasi Halaman
st.set_page_config(
    page_title="Spotify Track Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1DB954;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎵 Spotify Track Popularity Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi popularitas lagu menggunakan Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.markdown("---")

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    
    # Hapus kolom tidak relevan
    cols_to_drop = ['track_id', 'track_name', 'artist_name', 
                    'artist_genres', 'album_id', 'album_name']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Hapus missing values
    df = df.dropna()
    
    # Ubah Boolean ke int
    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(int)
    
    # Extract tahun rilis
    if 'album_release_date' in df.columns:
        df['release_year'] = pd.to_datetime(df['album_release_date'], errors='coerce').dt.year
        df = df.drop(columns=['album_release_date'])
    
    # Drop NA hasil konversi
    df = df.dropna()
    
    # One-Hot Encoding
    if 'album_type' in df.columns:
        df = pd.get_dummies(df, columns=['album_type'], drop_first=True)
    
    return df

# Fungsi untuk melatih model
@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    results = {}
    
    # Random Forest Pipeline
    pipe_rf = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('model', RandomForestRegressor(random_state=28))
    ])
    
    param_grid_rf = [
        {
            'feature_selection': [None],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5]
        },
        {
            'feature_selection': [SelectKBest(score_func=f_regression)],
            'feature_selection__k': [10, 20, 30],
            'model__n_estimators': [100],
            'model__max_depth': [10, 20],
        }
    ]
    
    with st.spinner('🔄 Melatih Random Forest Model...'):
        grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        y_pred_rf = grid_rf.best_estimator_.predict(X_test)
        
        results['Random Forest'] = {
            'model': grid_rf.best_estimator_,
            'best_params': grid_rf.best_params_,
            'cv_score': grid_rf.best_score_,
            'predictions': y_pred_rf,
            'mse': mean_squared_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'r2': r2_score(y_test, y_pred_rf)
        }
    
    # Decision Tree Pipeline
    pipe_dt = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('model', DecisionTreeRegressor(random_state=28))
    ])
    
    param_grid_dt = [
        {
            'feature_selection': [None],
            'model__max_depth': [3, 5, 8, 12, None],
            'model__min_samples_split': [2, 10, 20]
        },
        {
            'feature_selection': [SelectKBest(score_func=f_regression)],
            'feature_selection__k': [10, 20, 30],
            'model__max_depth': [10, 20],
            'model__min_samples_split': [2, 10]
        }
    ]
    
    with st.spinner('🔄 Melatih Decision Tree Model...'):
        grid_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=5, scoring='r2', n_jobs=-1)
        grid_dt.fit(X_train, y_train)
        y_pred_dt = grid_dt.best_estimator_.predict(X_test)
        
        results['Decision Tree'] = {
            'model': grid_dt.best_estimator_,
            'best_params': grid_dt.best_params_,
            'cv_score': grid_dt.best_score_,
            'predictions': y_pred_dt,
            'mse': mean_squared_error(y_test, y_pred_dt),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
            'mae': mean_absolute_error(y_test, y_pred_dt),
            'r2': r2_score(y_test, y_pred_dt)
        }
    
    return results

# Upload File
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    # Load dan Process Data
    with st.spinner('🔄 Memproses data...'):
        df_spotify = load_and_process_data(uploaded_file)
    
    st.success(f"✅ Data berhasil dimuat! Shape: {df_spotify.shape}")
    
    # Tab Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Eksplorasi Data", 
        "🤖 Model Training", 
        "📈 Evaluasi Model",
        "🔮 Prediksi",
        "📉 Visualisasi"
    ])
    
    # TAB 1: Eksplorasi Data
    with tab1:
        st.header("📊 Eksplorasi Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris", df_spotify.shape[0])
        with col2:
            st.metric("Total Kolom", df_spotify.shape[1])
        with col3:
            st.metric("Missing Values", df_spotify.isnull().sum().sum())
        
        st.subheader("Preview Data")
        st.dataframe(df_spotify.head(10), use_container_width=True)
        
        st.subheader("Statistik Deskriptif")
        st.dataframe(df_spotify.describe(), use_container_width=True)
        
        st.subheader("Distribusi Target (Track Popularity)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_spotify['track_popularity'], bins=50, color='#1DB954', edgecolor='black')
        ax.set_xlabel('Popularity Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribusi Track Popularity')
        st.pyplot(fig)
    
    # TAB 2: Model Training
    with tab2:
        st.header("🤖 Model Training")
        
        # Split Data
        target = 'track_popularity'
        X = df_spotify.drop(columns=[target])
        y = df_spotify[target]
        
        test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=28
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📚 Training Set: {X_train.shape[0]} samples")
        with col2:
            st.info(f"🧪 Test Set: {X_test.shape[0]} samples")
        
        if st.button("🚀 Mulai Training", type="primary"):
            start_time = time.time()
            results = train_models(X_train, y_train, X_test, y_test)
            end_time = time.time()
            
            st.success(f"✅ Training selesai dalam {end_time - start_time:.2f} detik!")
            
            # Simpan hasil ke session state
            st.session_state['results'] = results
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
    
    # TAB 3: Evaluasi Model
    with tab3:
        st.header("📈 Evaluasi Model")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Perbandingan Model
            st.subheader("Perbandingan Performa Model")
            
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': f"{metrics['r2']:.4f}",
                    'RMSE': f"{metrics['rmse']:.4f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'CV Score': f"{metrics['cv_score']:.4f}"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Detail per Model
            for model_name, metrics in results.items():
                with st.expander(f"📊 Detail {model_name}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with col3:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                    with col4:
                        st.metric("CV Score", f"{metrics['cv_score']:.4f}")
                    
                    st.write("**Best Parameters:**")
                    st.json(metrics['best_params'])
        else:
            st.warning("⚠️ Silakan lakukan training terlebih dahulu di tab 'Model Training'")
    
    # TAB 4: Prediksi
    with tab4:
        st.header("🔮 Prediksi Popularitas Lagu")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            model_choice = st.selectbox("Pilih Model", list(results.keys()))
            
            st.subheader("Prediksi vs Aktual (20 Sampel Pertama)")
            
            model = results[model_choice]['model']
            y_pred_sample = model.predict(X_test[:20])
            y_true_sample = y_test[:20].values if hasattr(y_test[:20], 'values') else y_test[:20]
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(20)
            ax.bar(x_pos - 0.2, y_true_sample, width=0.4, label='Actual', alpha=0.8, color='#1DB954')
            ax.bar(x_pos + 0.2, y_pred_sample, width=0.4, label='Predicted', alpha=0.8, color='#FF6B6B')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Popularity Score')
            ax.set_title(f'{model_choice}: Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            
            # Tabel Perbandingan
            df_pred = pd.DataFrame({
                'Sample': range(20),
                'Actual': y_true_sample,
                'Predicted': y_pred_sample,
                'Error': np.abs(y_true_sample - y_pred_sample)
            })
            st.dataframe(df_pred, use_container_width=True)
        else:
            st.warning("⚠️ Silakan lakukan training terlebih dahulu di tab 'Model Training'")
    
    # TAB 5: Visualisasi
    with tab5:
        st.header("📉 Visualisasi Analisis")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            # Residual Plots
            st.subheader("Residual Plots")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for idx, (name, metrics) in enumerate(results.items()):
                model = metrics['model']
                y_pred = model.predict(X_test)
                residuals = y_test.values - y_pred if hasattr(y_test, 'values') else y_test - y_pred
                
                axes[idx].scatter(y_pred, residuals, alpha=0.5, color='#1DB954')
                axes[idx].axhline(y=0, color='red', linestyle='--', linewidth=2)
                axes[idx].set_xlabel('Predicted Values')
                axes[idx].set_ylabel('Residuals')
                axes[idx].set_title(f'{name} - Residual Plot')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature Importance (Random Forest)
            if 'Random Forest' in results:
                st.subheader("Feature Importance (Random Forest)")
                
                rf_model = results['Random Forest']['model']
                # Cek apakah ada feature selection
                if hasattr(rf_model.named_steps['feature_selection'], 'get_support'):
                    # Ada feature selection
                    selected_features_mask = rf_model.named_steps['feature_selection'].get_support()
                    selected_features = X_test.columns[selected_features_mask]
                    importances = rf_model.named_steps['model'].feature_importances_
                else:
                    # Tidak ada feature selection
                    selected_features = X_test.columns
                    importances = rf_model.named_steps['model'].feature_importances_
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indices)), importances[indices], color='#1DB954')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([selected_features[i] for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top 10 Feature Importance')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("⚠️ Silakan lakukan training terlebih dahulu di tab 'Model Training'")

else:
    st.info("👆 Silakan upload file CSV dataset Spotify di sidebar untuk memulai")
    
    st.markdown("""
    ### 📝 Panduan Penggunaan:
    1. **Upload Dataset**: Upload file CSV dataset Spotify di sidebar
    2. **Eksplorasi Data**: Lihat statistik dan distribusi data
    3. **Model Training**: Klik tombol 'Mulai Training' untuk melatih model
    4. **Evaluasi Model**: Lihat perbandingan performa model
    5. **Prediksi**: Lihat hasil prediksi vs nilai aktual
    6. **Visualisasi**: Analisis residual dan feature importance
    
    ### 📊 Dataset yang Dibutuhkan:
    File CSV dengan kolom-kolom seperti:
    - `track_popularity` (target)
    - Audio features (danceability, energy, loudness, dll.)
    - Metadata (explicit, album_release_date, album_type, dll.)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Info")
st.sidebar.info("""
**Spotify Popularity Predictor**  
Developed with ❤️ using Streamlit  
Models: Random Forest & Decision Tree
""")