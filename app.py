import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import tempfile
import os
from All_functions_analysis import analyze_badminton_video 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Badminton Players Performance Analysis",
    page_icon="🏸",
    layout="wide"
)

# Load CSS from external file
def load_css(file_name):
    with open(file_name, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load the CSS
load_css("styles.css")





# Initialize session state
if "detection_complete" not in st.session_state:
    st.session_state.detection_complete = False


# #########################################################################################
st.title("🏸 Badminton Video Analysis")

# --------------------------------------------------------------------------------
# st.sidebar.image(r"images\img.png", width=200)

mode = st.sidebar.radio("Select Mode", ("Video Upload", "Live Camera"))


if mode == 'Video Upload':
# Upload video file
    uploaded_file = st.sidebar.file_uploader("Upload a badminton match video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        # Save uploaded file temporarily
        temp_dir = tempfile.NamedTemporaryFile(delete=False)
        input_video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(input_video_path)  # Show uploaded video

        # Define model paths
        model_paths = {
            'player': "model/players_detection_best.pt",
            'shuttle': "model/Shuttlecock_detection_best.pt",
            'shot': "model/shotType_detection_best.pt",
            'pose': "model/Pose_classification_best.pt"
        }

        # Define output file paths
        # Get current working directory
        current_dir = os.getcwd()  

        # Define the "output" directory inside the current working directory
        output_dir = os.path.join(current_dir, "output")

        # Ensure the "output" directory exists
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Define the output file paths
        stats_output_paths = {
            "output_path": os.path.join(output_dir, "output_video.mp4"),
            "player_stats": os.path.join(output_dir, "player_stats.csv"),
            "player1_shots": os.path.join(output_dir, "player1_shots.csv"),
            "player2_shots": os.path.join(output_dir, "player2_shots.csv"),
            "player1_poses": os.path.join(output_dir, "player1_poses.csv"),
            "player2_poses": os.path.join(output_dir, "player2_poses.csv"),
        }
        
        # Run analysis function when button is clicked
        if st.sidebar.button("Analyze Video"):
            with st.spinner("Processing video... ⏳"):
                analyze_badminton_video(input_video_path, model_paths, stats_output_paths)

            st.session_state.detection_complete = True
            st.sidebar.success("✅ Analysis Complete!")

# --------------------------------------- TRUE
if st.session_state.detection_complete:

    # Function to read CSV if it exists
    def read_csv_if_exists(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found. Returning empty DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist

    # Read player data safely
    p1poses = read_csv_if_exists("output\player1_poses.csv")
    p1shots = read_csv_if_exists("output\player1_shots.csv")
    p2poses = read_csv_if_exists("output\player2_poses.csv")
    p2shots = read_csv_if_exists("output\player2_shots.csv")

    player_stats = read_csv_if_exists("output\player_stats.csv")


# with Dashboard_tab2: ---------------------------------------------------------------------------------------------------------------------------------------------------
    st.title("🏸 Badminton Video Analysis")

    player1, player2 = st.columns(2)
    

    with player1: #-----------------------------------------------
        st.title("Player 1 Performance")

        ForehandLift_count = p1poses["Pose"].str.count("ForehandLift").sum()

        BackhandLift_count = p1poses["Pose"].str.count("BackhandLift").sum()

        Smash_count = (p1poses["Pose"].str.count("Smash").sum())
        
        p1_speed_avg = int(player_stats["Player 1 Speed"].mean())
        
        p1_speed_max = int(player_stats["Player 1 Speed"].max())

        # Wrap all metrics inside a single div with custom styling
        
        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-item">
                <div class="label">ForehandLift count</div>
                <div class="value">{ForehandLift_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">BackhandLift count</div>
                <div class="value">{BackhandLift_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">Smash count</div>
                <div class="value">{Smash_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">⚡Avg speed</div>
                <div class="value">{p1_speed_avg}</div>
            </div>
            <div class="metric-item">
                <div class="label">🚀Max speed</div>
                <div class="value">{p1_speed_max}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
        
        # -----------------------------------------------------------------------
        plot1, plot2 = st.columns(2)

        with plot1:
            # Count occurrences of each Shot_Type
            shot_counts = p1shots["Shot_Type"].value_counts().reset_index()
            shot_counts.columns = ["Shot_Type", "Count"]
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
            
            # Bar Chart 📊
            fig = px.bar(
                shot_counts, x="Shot_Type", y="Count", 
                color="Shot_Type",
                text="Count",
                color_discrete_sequence=colors,
                title="Shot Distribution"
            )
            # Remove background
            fig.update_layout(
                title_font=dict(color='white'),
                legend=dict(font=dict(color='white')),
                xaxis_title="Shot Type",
                yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                xaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with plot2:
            # Pie Chart 
            fig2 = px.pie(
                shot_counts, names="Shot_Type", values="Count",
                color_discrete_sequence=colors, title="Shot Distribution (%)"
            )
            # Remove background
            fig2.update_layout(
                title_font=dict(color='white'),
                legend=dict(font=dict(color='white')),
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
                paper_bgcolor="rgba(0,0,0,0)"   # Transparent paper background
            )

            fig2.update_traces(textinfo="percent+label", pull=[0.05] * len(shot_counts))
            st.plotly_chart(fig2, use_container_width=True)



    with player2: # ------------------------------------------------

        st.title("Player 2 Performance")

        ForehandLift_count = p2poses["Pose"].str.count("ForehandLift").sum()

        BackhandLift_count = p2poses["Pose"].str.count("BackhandLift").sum()
    
        Smash_count = (p2poses["Pose"].str.count("Smash").sum())

        p2_speed_avg = int(player_stats["Player 2 Speed"].mean())

        p2_speed_max = int(player_stats["Player 2 Speed"].max())


        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-item">
                <div class="label">ForehandLift count</div>
                <div class="value">{ForehandLift_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">BackhandLift count</div>
                <div class="value">{BackhandLift_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">Smash count</div>
                <div class="value">{Smash_count}</div>
            </div>
            <div class="metric-item">
                <div class="label">⚡Avg speed</div>
                <div class="value">{p2_speed_avg}</div>
            </div>
            <div class="metric-item">
                <div class="label">🚀Max speed</div>
                <div class="value">{p2_speed_max}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
        
        
        # ---------------------------------------------------------
        plot1, plot2 = st.columns(2)

        with plot1:
            # Count occurrences of each Shot_Type
            shot_counts = p2shots["Shot_Type"].value_counts().reset_index()
            shot_counts.columns = ["Shot_Type", "Count"]
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

            # Bar Chart 📊
            fig = px.bar(
                shot_counts, x="Shot_Type", y="Count", 
                color="Shot_Type",
                text="Count",
                color_discrete_sequence=colors,
                title="Shot Distribution"
            )
             # Remove background color
            fig.update_layout(
                title_font=dict(color='white'),
                legend=dict(font=dict(color='white')),
                xaxis_title="Shot Type",
                yaxis_title="Count",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                xaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with plot2:
            # Pie Chart 
            fig2 = px.pie(
                shot_counts, 
                names="Shot_Type", 
                values="Count",
                color_discrete_sequence=colors, 
                title="Shot Distribution (%)"
            )
            # Remove background color
            fig2.update_layout(
                title_font=dict(color='white'),
                legend=dict(font=dict(color='white')),
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
                paper_bgcolor="rgba(0,0,0,0)"  # Transparent paper background
            )
            fig2.update_traces(textinfo="percent+label", pull=[0.05] * len(shot_counts))
            st.plotly_chart(fig2, use_container_width=True)

    #--------------------------------------------------------------------------------------------
    p1_p2, = st.columns(1)  
    with p1_p2:
            
        # Streamlit UI
        st.title("🏸 Badminton Player Speeds Over Time")

        # Create a Line Chart with Plotly
        fig = px.line(
            player_stats, 
            x="Frame", 
            y=["Player 1 Speed", "Player 2 Speed"], 
            title="P1 & P2 Speeds Over Time"
        )

        # Remove Background
        fig.update_layout(
            title_font=dict(color='white'),
            legend=dict(font=dict(color='white')),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
            xaxis=dict(showgrid=False),  # Hide x-axis grid
            yaxis=dict(showgrid=False)   # Hide y-axis grid
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
