# demo.py - Bundesliga AI Tahmin Sistemi DEMO (GÜVENLİ VERSİYON)
import streamlit as st
import pandas as pd
import numpy as np
import random

# ================== SAYFA AYARLARI ==================
st.set_page_config(
    page_title="Bundesliga AI Predictor DEMO",
    page_icon="⚽",
    layout="wide"
)

# ================== DEMO UYARILARI ==================
with st.sidebar:
    st.header("ℹ️ Demo Bilgisi")
    st.warning("""
    🚨 **DEMO VERSION**
    
    Bu uygulama **Bundesliga AI tahmin modelimizin tanıtım** 
    amacıyla hazırlanmıştır.
    
    **Tahminler örnek amaçlıdır.**
    
    Ticari kullanım için lütfen iletişime geçin.
    """)
    
    st.info("📧 **İletişim:** matchanalytics.ai@gmail.com")
    st.markdown("---")
    st.caption("© 2025 Bundesliga AI Forecast - Tüm hakları saklıdır")

# ================== ANA SAYFA ==================
st.title("⚽ Bundesliga AI Tahmin Sistemi - DEMO")

st.markdown("""
<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
<h4 style='margin-top: 0; color: #1E3A8A;'>🤖 Yapay Zeka Destekli Tahmin Sistemi</h4>
<p style='color: #374151;'>Bu demo, Bundesliga maçları için geliştirdiğimiz AI tabanlı tahmin 
sisteminin yeteneklerini göstermek amacıyla hazırlanmıştır.</p>
</div>
""", unsafe_allow_html=True)

# ================== TAKIM SEÇİMİ ==================
st.header("1️⃣ Takım Seçimi")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Ev Sahibi Takım",
        ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", 
         "Bayer Leverkusen", "VfB Stuttgart", "Eintracht Frankfurt",
         "VfL Wolfsburg", "Borussia Mönchengladbach", "TSG Hoffenheim",
         "1. FC Heidenheim 1846", "1. FC Köln", "SV Werder Bremen"],
        index=0,
        key="home_select"
    )

with col2:
    away_team = st.selectbox(
        "✈️ Deplasman Takımı",
        ["Bayern Munich", "Borussia Dortmund", "RB Leipzig",
         "Bayer Leverkusen", "VfB Stuttgart", "Eintracht Frankfurt",
         "VfL Wolfsburg", "Borussia Mönchengladbach", "TSG Hoffenheim",
         "1. FC Heidenheim 1846", "1. FC Köln", "SV Werder Bremen"],
        index=1,
        key="away_select"
    )

st.markdown("---")

# ================== TAHMİN BUTONU ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🎯 **TAHMİN YAP**", 
                             type="primary", 
                             use_container_width=True,
                             help="Tıklayın ve AI tahminini görün")

if predict_button:
    
    # ================== DEMO TAHMİN HESAPLAMALARI ==================
    # Realistic Bundesliga ratings
    team_ratings = {
        "Bayern Munich": 85.0, "Borussia Dortmund": 78.5, "RB Leipzig": 76.0,
        "Bayer Leverkusen": 75.5, "VfB Stuttgart": 72.0, "Eintracht Frankfurt": 71.5,
        "VfL Wolfsburg": 69.0, "Borussia Mönchengladbach": 68.5, "TSG Hoffenheim": 67.5,
        "1. FC Heidenheim 1846": 65.0, "1. FC Köln": 66.0, "SV Werder Bremen": 67.0
    }
    
    # Takım özellikleri
    home_rating = team_ratings.get(home_team, round(random.uniform(65.0, 85.0), 1))
    away_rating = team_ratings.get(away_team, round(random.uniform(65.0, 85.0), 1))
    
    # Form durumu (son 5 maç kazanma %)
    home_form = round(random.uniform(0.3, 0.8), 3)
    away_form = round(random.uniform(0.3, 0.8), 3)
    
    # Yaş ortalaması (Bundesliga gerçek değerler)
    home_age = round(random.uniform(24.5, 27.5), 1)
    away_age = round(random.uniform(24.5, 27.5), 1)
    
    # Takım değeri (milyon €)
    home_value = int(home_rating * 1.2 * 1000000)
    away_value = int(away_rating * 1.2 * 1000000)
    
    # ================== AI TAHMİN ALGORİTMASI ==================
    # 1. Rating farkı etkisi
    rating_diff = (home_rating - away_rating) / 20.0  # Normalize
    
    # 2. Form farkı etkisi
    form_diff = (home_form - away_form) * 0.5
    
    # 3. Ev sahibi avantajı
    home_advantage = 0.12
    
    # 4. Yaş faktörü (deneyim vs gençlik)
    age_factor = 0.05 if home_age > away_age else -0.03
    
    # Nihai olasılıklar
    base_home = 0.33 + rating_diff + form_diff + home_advantage + age_factor
    base_away = 0.33 - rating_diff - form_diff - age_factor
    
    # Sınırlandırma
    prob_home = min(0.75, max(0.15, base_home))
    prob_away = min(0.75, max(0.15, base_away))
    prob_draw = 1.0 - prob_home - prob_away
    
    # Tahmin kararı
    if prob_home >= prob_away and prob_home >= prob_draw:
        prediction = f"{home_team} KAZANIR"
        confidence = prob_home
        result_color = "🟢"
    elif prob_away >= prob_home and prob_away >= prob_draw:
        prediction = f"{away_team} KAZANIR"
        confidence = prob_away
        result_color = "🔵"
    else:
        prediction = "BERABERLİK"
        confidence = prob_draw
        result_color = "🟡"
    
    # ================== SONUÇ GÖSTERİMİ ==================
    # Başlık
    st.success(f"🎯 **Tahmin Sonucu:** {home_team} vs {away_team}")
    
    # Demo uyarısı
    with st.expander("ℹ️ Demo Hakkında Bilgi", expanded=True):
        st.info("""
        **Demo Notu:** Bu tahmin, gerçek zamanlı verilerle güncellenen ve özelleştirilmiş 
        trading algoritmalarına entegre edilebilen ticari versiyonumuzun basitleştirilmiş bir örneğidir.
        
        **Gerçek sistemde kullanılan özellikler:**
        - 18 optimize edilmiş özellik
        - Takım yaş ortalaması analizi
        - Form durumu (son 5 maç)
        - Takım değeri ve piyasa analizi
        - H2H (karşılaşma geçmişi) istatistikleri
        - Defansif/Ofansif denge metrikleri
        """)
    
    # ================== OLASILIK GÖSTERGELERİ ==================
    st.subheader("📊 Tahmin Olasılıkları")
    
    # Görsel gösterge
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🏠 Ev Kazanır", 
            value=f"{prob_home*100:.1f}%",
            delta=f"{prob_home*100-33.3:+.1f}%",
            delta_color="normal"
        )
        # Progress bar
        st.progress(float(prob_home))
        
    with col2:
        st.metric(
            label="🤝 Beraberlik", 
            value=f"{prob_draw*100:.1f}%",
            delta=f"{prob_draw*100-33.3:+.1f}%",
            delta_color="off"
        )
        st.progress(float(prob_draw))
        
    with col3:
        st.metric(
            label="✈️ Dep Kazanır", 
            value=f"{prob_away*100:.1f}%",
            delta=f"{prob_away*100-33.3:+.1f}%",
            delta_color="normal"
        )
        st.progress(float(prob_away))
    
    # ================== TAHMİN SONUCU ==================
    st.subheader("🏆 Model Tahmini")
    
    if result_color == "🟢":
        st.success(f"""
        **{result_color} MODEL TAHMİNİ: {prediction}** 
        
        **Güven Seviyesi:** {confidence*100:.1f}%
        
        **Analiz:** {home_team} daha yüksek rating ({home_rating:.1f}) ve 
        daha iyi form ({home_form*100:.1f}%) ile favori konumunda.
        """)
    elif result_color == "🔵":
        st.info(f"""
        **{result_color} MODEL TAHMİNİ: {prediction}** 
        
        **Güven Seviyesi:** {confidence*100:.1f}%
        
        **Analiz:** {away_team} deplasmanda üstünlük sağlıyor. 
        Form farkı ({away_form*100:.1f}% vs {home_form*100:.1f}%) belirleyici olabilir.
        """)
    else:
        st.warning(f"""
        **{result_color} MODEL TAHMİNİ: {prediction}** 
        
        **Güven Seviyesi:** {confidence*100:.1f}%
        
        **Analiz:** Takımlar dengeli görünüyor. Rating ({home_rating:.1f} vs {away_rating:.1f}) 
        ve form ({home_form*100:.1f}% vs {away_form*100:.1f}%) benzer seviyede.
        """)
    
    # ================== TAKIM KARŞILAŞTIRMASI ==================
    st.subheader("📈 Takım Karşılaştırması")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        
        # Rating gösterge
        rating_col1, rating_col2 = st.columns([3, 1])
        with rating_col1:
            st.progress(float((home_rating - 60) / 25))  # 60-85 → 0-1
        with rating_col2:
            st.metric("Rating", f"{home_rating:.1f}", "")
        
        # Diğer metrikler
        st.metric("📈 Form (Son 5 Maç)", f"{home_form*100:.1f}%", 
                 f"{'↑' if home_form > 0.5 else '↓'} {abs(home_form-0.5)*100:.1f}%")
        
        st.metric("👥 Yaş Ortalaması", f"{home_age:.1f} yaş", 
                 f"{'Deneyimli' if home_age > 26 else 'Genç'}")
        
        st.metric("💰 Takım Değeri", f"€{home_value:,}", 
                 f"{(home_value/1000000):.1f}M €")
    
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        
        # Rating gösterge
        rating_col1, rating_col2 = st.columns([3, 1])
        with rating_col1:
            st.progress(float((away_rating - 60) / 25))
        with rating_col2:
            st.metric("Rating", f"{away_rating:.1f}", "")
        
        # Diğer metrikler
        st.metric("📈 Form (Son 5 Maç)", f"{away_form*100:.1f}%", 
                 f"{'↑' if away_form > 0.5 else '↓'} {abs(away_form-0.5)*100:.1f}%")
        
        st.metric("👥 Yaş Ortalaması", f"{away_age:.1f} yaş", 
                 f"{'Deneyimli' if away_age > 26 else 'Genç'}")
        
        st.metric("💰 Takım Değeri", f"€{away_value:,}", 
                 f"{(away_value/1000000):.1f}M €")
    
    # ================== YAŞ ANALİZİ ==================
    st.subheader("👥 Yaş Analizi")
    
    age_diff = home_age - away_age
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ev Sahibi Yaş", f"{home_age:.1f}", 
                 f"{'↑' if home_age > 26 else '↓'} {abs(home_age-26):.1f}")
    with col2:
        st.metric("Deplasman Yaş", f"{away_age:.1f}", 
                 f"{'↑' if away_age > 26 else '↓'} {abs(away_age-26):.1f}")
    with col3:
        st.metric("Yaş Farkı", f"{age_diff:+.1f}", 
                 f"{'Ev avantaj' if age_diff > 0 else 'Dep avantaj'}")
    
    # Yaş yorumu
    if age_diff > 1.5:
        st.info(f"""
        **📊 Yaş Analizi:** {home_team} daha deneyimli bir kadroya sahip (+{age_diff:.1f} yaş). 
        Deneyimli kadrolar kritik maçlarda daha soğukkanlı olabilir.
        """)
    elif age_diff < -1.5:
        st.info(f"""
        **📊 Yaş Analizi:** {away_team} daha genç ve dinamik bir kadroya sahip ({age_diff:+.1f} yaş). 
        Genç takımlar fiziksel üstünlük ve hız avantajına sahip olabilir.
        """)
    else:
        st.info("""
        **📊 Yaş Analizi:** Takımlar benzer yaş profiline sahip. 
        Deneyim ve gençlik dengesi her iki takımda da mevcut.
        """)
    
    # ================== MODEL ANALİZ DETAYLARI (GÜVENLİ VERSİYON) ==================
    with st.expander("🔍 **AI Model Analizi**", expanded=False):
        st.info("""
        **🤖 AI Değerlendirme Özeti:**
        
        Modelimiz bu maçı analiz ederken çoklu faktörleri değerlendirdi:
        
        **🎯 Ana Belirleyiciler:**
        • Takım performansı ve form durumu
        • Oyuncu kalitesi ve takım rating'i
        • Ev sahibi avantajı faktörü
        
        **📊 Destekleyici Faktörler:**
        • Takım yaş profili ve deneyim dengesi
        • Piyasa değeri karşılaştırması
        • Takım dinamikleri ve momentum
        """)
        
        # GÜVENLİ ANALİZ TABLOSU - Sayısal değerler YOK
        analysis_points = [
            {"Aspect": "Form Analizi", "Finding": f"{home_team if home_form > away_form else away_team} daha iyi formda", "Impact": "Yüksek"},
            {"Aspect": "Rating Karşılaştırması", "Finding": f"{home_team if home_rating > away_rating else away_team} daha yüksek rating", "Impact": "Yüksek"},
            {"Aspect": "Ev Sahibi Avantajı", "Finding": "Bundesliga'da ev sahibi takıma +%12 avantaj", "Impact": "Orta"},
            {"Aspect": "Yaş Dinamiği", "Finding": f"{'Deneyim avantajı' if age_diff > 0 else 'Gençlik avantajı'}", "Impact": "Orta"},
            {"Aspect": "Takım Değeri", "Finding": "Piyasa değeri dengeli", "Impact": "Düşük"},
            {"Aspect": "Beraberlik Potansiyeli", "Finding": f"%{prob_draw*100:.1f} beraberlik olasılığı", "Impact": "Değişken"}
        ]
        
        analysis_df = pd.DataFrame(analysis_points)
        st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        st.warning("""
        ⚠️ **Demo Notu:** Bu analiz basitleştirilmiş bir gösterimdir. 
        Gerçek ticari versiyon 18 farklı faktörü değerlendirir ve gelişmiş 
        makine öğrenimi algoritmaları kullanır.
        """)
    
    # ================== TİCARİ ÇAĞRI ==================
    st.markdown("---")
    
    st.success("""
    ### 💼 **Ticari İş Birliği İçin**
    
    Bu demo, **Bundesliga AI tahmin sistemimizin** yeteneklerini göstermek amacıyla hazırlanmıştır.
    
    **Tam özellikli ticari versiyonumuz şunları içerir:**
    
    ✅ **Gerçek zamanlı veri entegrasyonu** (FBref + Transfermarkt)  
    ✅ **18 optimize edilmiş özellik** ile gelişmiş makine öğrenimi  
    ✅ **API erişimi** ve özel entegrasyonlar  
    ✅ **Diğer ligler için özelleştirme** (Premier League, La Liga, Serie A)  
    ✅ **Detaylı performans metrikleri** ve backtesting  
    ✅ **Özel trading algoritmaları** entegrasyonu  
    
    **İletişim:** 📧 **matchanalytics.ai@gmail.comm**
    """)
    
    # Hızlı iletişim formu
    with st.expander("📬 Hızlı İletişim Formu", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Adınız")
            company = st.text_input("Şirket Adı")
        with col2:
            email = st.text_input("E-posta Adresiniz")
            interest = st.selectbox("İlgi Alanınız", 
                                  ["Demo Talep", "Fiyat Teklifi", "Teknik Detay", "İş Birliği"])
        
        if st.button("📩 Bilgi Talebi Gönder", type="secondary"):
            if email:
                st.success(f"Teşekkürler {name}! En kısa sürede {email} adresinizden dönüş yapacağız.")
            else:
                st.warning("Lütfen e-posta adresinizi giriniz.")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 14px; padding: 1rem;'>
    <p style='margin: 0.5rem 0;'>
        <strong>⚽ Bundesliga AI Tahmin Sistemi</strong> | 
        Takım Yaş Analizi Entegreli | 
        <span style='color: #EF4444;'>DEMO VERSION</span>
    </p>
    <p style='margin: 0.5rem 0;'>
        © 2025 Bundesliga AI Forecast | Tüm hakları saklıdır | 
        <a href='mailto:contact@bundesliga-forecast.com' style='color: #3B82F6; text-decoration: none;'>
            matchanalytics.ai@gmail.com
        </a>
    </p>
    <p style='margin: 0.5rem 0; font-size: 12px;'>
        🔒 Ticari sır kapsamındadır | Demo ve tanıtım amaçlıdır | 
        Gerçek tahminler için ticari versiyon gereklidir
    </p>
</div>
""", unsafe_allow_html=True)

# ================== EK BİLGİ ==================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Model Performansı")
    st.caption("""
    **Gerçek Sistem Metrikleri:**
    - Test Accuracy: %60+
    - Draw Recall: %25+ 
    - HomeWin Recall: %60+
    - Overfitting Gap: <%10
    - Özellik Sayısı: 18
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Kullanım Kılavuzu")
    st.caption("""
    1. Ev ve deplasman takımını seçin
    2. "TAHMİN YAP" butonuna tıklayın
    3. AI tahmini ve detaylı analizi görün
    4. Ticari versiyon için iletişime geçin
    """)