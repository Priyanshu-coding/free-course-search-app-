import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import os
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
import torch
from fuzzywuzzy import process

# Load data from CSV
file_path = 'free_courses_data.csv'  # The path to your CSV file
try:
    courses_data = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"CSV file not found at: {file_path}")

# Load the Analytics Vidhya logo from the images folder in the project
logo_path = 'Images/analytics_vidhya_logo.jpeg'


# Function to resize image
def resize_image(image_path, size=(300, 300)):
    try:
        image = Image.open(image_path)
        resized_image = ImageOps.fit(image, size, Image.LANCZOS)
        return resized_image
    except FileNotFoundError:
        st.error(f"Image file not found at {image_path}")
        return None


# Function to convert image to base64 for displaying
def get_image_base64(image_path):
    try:
        img = Image.open(image_path)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except FileNotFoundError:
        st.error(f"File not found at {image_path}")
        return ""


# Function to display categories with images and clickable category names
def display_categories_with_clickable_names():
    categories = courses_data['Category'].unique()
    cols = st.columns(2)
    for i, category in enumerate(categories):
        with cols[i % 2]:
            image_path = f'Images/{category}.jpeg'
            if os.path.exists(image_path):
                resized_image = resize_image(image_path)
                if resized_image:
                    st.image(resized_image, use_column_width=True)
                if st.button(category):
                    st.session_state.selected_category = category
            else:
                st.error(f"Image for {category} not found.")


# Function to display courses under the selected category
def display_courses_in_category(category):
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F28D8C;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("Back to All Courses"):
        st.session_state.selected_category = None

    st.markdown(f"## {category} Courses")
    courses_in_category = courses_data[courses_data['Category'] == category]
    for _, course in courses_in_category.iterrows():
        st.markdown(f"### [{course['Course Name']}]({course['Link']})")
        st.write(course['Description'])
    st.write("---")


# Load the pre-trained sentence-transformers model for semantic search
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


# Function to provide real-time course suggestions as the user types
def suggest_courses(user_input):
    course_names = courses_data['Course Name'].tolist()
    suggestions = process.extract(user_input, course_names, limit=5)
    return [sug[0] for sug in suggestions]


# Enhanced search function with a comprehensive guide and structured learning paths
def search_courses(query):
    st.markdown("## Search Results")

    # Search term matches specific topics
    if "python" in query.lower():
        display_course_path("Python")
        suggest_next_steps("python")

    elif "business analytics" in query.lower():
        display_course_path("Business Analytics")
        suggest_next_steps("business analytics")

    elif "machine learning" in query.lower():
        display_course_path("Machine Learning")
        suggest_next_steps("machine learning")

    elif "deep learning" in query.lower():
        display_course_path("Deep Learning")
        suggest_next_steps("deep learning")

    elif "generative ai" in query.lower():
        display_course_path("Generative AI")
        suggest_next_steps("generative ai")

    elif "nlp" in query.lower():
        display_course_path("NLP")
        suggest_next_steps("nlp")

    else:
        # Perform general semantic search if no specific keyword matches
        query_embedding = model.encode(query, convert_to_tensor=True)
        course_descriptions = courses_data['Description'].tolist()
        course_embeddings = model.encode(course_descriptions, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, course_embeddings)
        filtered_courses = pd.DataFrame({
            'Course Name': courses_data['Course Name'],
            'Link': courses_data['Link'],
            'Description': courses_data['Description'],
            'Similarity': similarities.tolist()
        }).sort_values(by='Similarity', ascending=False)

        # Display top 5 courses based on similarity
        for _, course in filtered_courses.head(5).iterrows():
            st.markdown(f"### [{course['Course Name']}]({course['Link']})")
            st.write(course['Description'])
            st.write(f"**Similarity:** {course['Similarity']:.4f}")
            st.write("---")


# Function to display a structured course path for different topics
def display_course_path(topic):
    courses_in_topic = courses_data[courses_data['Category'].str.contains(topic, case=False)]
    st.markdown(f"### Here are some {topic} courses to help you get started:")
    for _, course in courses_in_topic.iterrows():
        st.markdown(f"### [{course['Course Name']}]({course['Link']})")
        st.write(course['Description'])
        st.write("---")


# Function to suggest further learning paths for different topics
def suggest_next_steps(topic):
    if topic.lower() == "python":
        st.markdown("""
            **After mastering Python, you should explore:**
            - Machine Learning (ML)
            - Deep Learning
            - AI Development
            Python is foundational for these fields, and it's in high demand for data science roles.
        """)

    elif topic.lower() == "business analytics":
        st.markdown("""
            **In Business Analytics, you can next learn:**
            - Data Visualization (using tools like Power BI, Tableau)
            - Predictive Analytics
            - Statistical Analysis
            Mastering these skills will help you drive data-driven decisions in business environments.
        """)

    elif topic.lower() == "machine learning":
        st.markdown("""
            **Before diving deep into Machine Learning, start with:**
            - Python Programming
            - Basic Statistics
            - Fundamentals of Machine Learning Algorithms
            Once you grasp these, move on to advanced ML techniques like Ensemble Learning and Neural Networks.
        """)

    elif topic.lower() == "deep learning":
        st.markdown("""
            **Before diving into Deep Learning, it's important to understand:**
            - Python (for coding skills)
            - Machine Learning (for understanding algorithms)
            - Business Analytics (for data handling and interpretation)
            Once you're familiar with these, dive into Deep Learning!
        """)

    elif topic.lower() == "generative ai":
        st.markdown("""
            **To excel in Generative AI, follow these steps:**
            - Learn Python and Machine Learning basics
            - Understand Deep Learning models
            - Explore Generative AI models like GPT, GANs, etc.
            Master these steps to gain expertise in the AI field.
        """)

    elif topic.lower() == "nlp":
        st.markdown("""
            **Natural Language Processing (NLP) requires:**
            - Python proficiency
            - Familiarity with text preprocessing
            - Understanding of basic Machine Learning techniques
            From there, delve into advanced NLP topics like sentiment analysis, text generation, and named entity recognition.
        """)


# Streamlit App Interface
def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #AA7CA7;
        }
        .logo {
            float: right;
            margin-right: 30px;
            margin-top: -40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and logo together
    logo_base64 = get_image_base64(logo_path)
    if logo_base64:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h1>Analytics Vidhya Free Courses</h1>
                <img src="data:image/jpeg;base64,{logo_base64}" class="logo" width="150">
            </div>
            """,
            unsafe_allow_html=True
        )

    # Search with suggestions
    user_input = st.text_input("Search for a course:", key="search_input")

    if user_input:
        suggestions = suggest_courses(user_input)
        st.write(f"Suggestions: {', '.join(suggestions)}")

        if st.button("Search"):
            search_courses(user_input)

    selected_category = st.session_state.get('selected_category')

    if not selected_category:
        st.header("All Courses")
        display_categories_with_clickable_names()
    else:
        display_courses_in_category(selected_category)


if __name__ == '__main__':
    if 'selected_category' not in st.session_state:
        st.session_state['selected_category'] = None

    main()
