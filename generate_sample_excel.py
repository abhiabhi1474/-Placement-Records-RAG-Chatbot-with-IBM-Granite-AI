"""
Run this once to regenerate placements.xlsx with sample data.
Replace with your real data when deploying.
"""
import pandas as pd

data = {
    "Name": [
        "Abhi Sharma", "Rahul Menon", "Priya Nair", "Arun Kumar",
        "Sneha Das", "Vikram Rao", "Anjali Pillai", "Deepak Singh",
    ],
    "Branch": ["AIML", "CSE", "ECE", "AIML", "CSE", "MECH", "AIML", "CSE"],
    "Graduation_Year": [2024, 2024, 2025, 2025, 2024, 2024, 2025, 2025],
    "CGPA": [8.9, 8.2, 7.8, 9.1, 8.5, 7.5, 9.3, 8.0],
    "Skills": [
        "Python, NLP, Machine Learning, LangChain",
        "Java, Spring Boot, Microservices, REST API",
        "Embedded C, VLSI, Signal Processing, MATLAB",
        "Deep Learning, Computer Vision, PyTorch, OpenCV",
        "React, Node.js, MongoDB, TypeScript",
        "AutoCAD, ANSYS, SolidWorks, FEA",
        "Generative AI, RAG, Hugging Face, Fine-tuning",
        "Python, Django, PostgreSQL, Docker",
    ],
    "Projects": [
        "Legal Document AI using RAG and IBM Granite",
        "Banking Microservices Platform with Spring Cloud",
        "Smart Home Automation using IoT and MQTT",
        "Real-time Object Detection for Autonomous Vehicles",
        "E-commerce Platform with React and Node.js",
        "Stress Analysis of Aircraft Components using FEA",
        "Placement Assistant Chatbot using LLM and FAISS",
        "Healthcare Management System with REST APIs",
    ],
    "Placed_Company": [
        "IBM Research", "TCS", "Qualcomm", "NVIDIA",
        "Infosys", "L&T", "Amazon AI", "Wipro",
    ],
    "Package_LPA": [18, 10, 14, 22, 9, 8, 24, 11],
}

df = pd.DataFrame(data)
df.to_excel("placements.xlsx", index=False)
print(f"Generated placements.xlsx with {len(df)} rows.")
print(df.to_string(index=False))
