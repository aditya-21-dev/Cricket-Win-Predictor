import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load model and encoder ---
try:
    model = joblib.load("win_predictor.pkl")
    team_encoder = joblib.load("team_encoder.pkl")
    print("Model and encoder loaded successfully.")
except FileNotFoundError:
    st.error("🚨 Model or encoder files not found! Please add 'win_predictor.pkl' and 'team_encoder.pkl' to the app's folder.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# Get the list of known teams from the encoder
team_list = team_encoder.classes_

st.set_page_config(page_title="Cricket Win Predictor", layout="wide")

st.markdown("<h2 style='text-align:center;'>🏏 LIVE CRICKET WIN PREDICTOR</h2>", unsafe_allow_html=True)
st.markdown("---")

# --- Input section ---
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team", team_list)
    runs = st.number_input("Current Score (Runs)", min_value=0, value=120)
    
    # Split overs and balls for easier input
    over_col, ball_col = st.columns(2)
    with over_col:
        overs = st.number_input("Overs Completed", min_value=0, max_value=19, value=15)
    with ball_col:
        balls = st.number_input("Balls Bowled (in current over)", min_value=0, max_value=5, value=0)

with col2:
    bowling_team = st.selectbox("Select Bowling Team", team_list)
    target = st.number_input("Target Runs", min_value=1, value=170)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3)

st.markdown("---")

# --- Validation ---
if batting_team == bowling_team:
    st.error("⚠️ Batting team and Bowling team cannot be the same!")
else:
    if st.button("🔮 Predict Win Probability"):

        # --- 1. Calculate Stateful Features ---
        try:
            balls_bowled = (overs * 6) + balls
            balls_left = 120 - balls_bowled
            
            runs_left = target - runs
            # Handle case where runs left is negative (team already won)
            if runs_left < 0:
                runs_left = 0
                
            wickets_left = 10 - wickets

            # --- 2. Calculate Run Rates ---
            # Handle division by zero
            crr = (runs * 6) / balls_bowled if balls_bowled > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else (float('inf') if runs_left > 0 else 0)

            # --- 3. Encode Teams ---
            bat_enc = team_encoder.transform([batting_team])[0]
            bowl_enc = team_encoder.transform([bowling_team])[0]

            # --- 4. Create Input DataFrame for Model ---
            # This MUST match the columns your model was trained on
            sample = pd.DataFrame([{
                "batting_team_encoded": bat_enc,
                "bowling_team_encoded": bowl_enc,
                "target": target,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "crr": crr,
                "rrr": rrr
            }])
            
            # --- 5. Get Prediction ---
            # model.predict_proba returns [[prob_loss, prob_win]]
            win_prob = model.predict_proba(sample)[0][1] # Get prob_win
            
            # (Safety check for RRR infinity)
            if rrr == float('inf'):
                win_prob = 0.0 
            
            team1_prob = round(win_prob * 100, 2)
            team2_prob = round(100 - team1_prob, 2)

            st.markdown(f"<h4 style='text-align:center;'>{batting_team} vs {bowling_team}</h4>", unsafe_allow_html=True)

            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"<h5 style='color:green;text-align:center;'>{batting_team}</h5>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:green;text-align:center;'>{team1_prob}%</h3>", unsafe_allow_html=True)
            with colB:
                st.markdown(f"<h5 style='color:blue;text-align:center;'>{bowling_team}</h5>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:blue;text-align:center;'>{team2_prob}%</h3>", unsafe_allow_html=True)

            st.progress(int(team1_prob))

            st.success(f"🏆 Predicted Winning Team: {batting_team if team1_prob > team2_prob else bowling_team}")

            st.caption(f"📊 Balls left: {balls_left} | Runs remaining: {runs_left} | Wickets left: {wickets_left} | RRR: {rrr:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
