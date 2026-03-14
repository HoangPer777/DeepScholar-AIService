import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env from absolute path
dotenv_path = r"d:\aaa\aa_study_document\a_year 4 semester 2\Graduation Project\DeepScholar\DeepScholar-AIService\.env"
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"Checking models for API key: {api_key[:10]}...")

try:
    print("\n--- v1 Models ---")
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"Name: {m.name}, Version: v1?")
            
    # Try listing via v1beta explicitly if possible by just listing again (SDK usually defaults)
    print("\n--- All Models ---")
    for m in genai.list_models():
        print(f"Name: {m.name}, Supported: {m.supported_generation_methods}")

except Exception as e:
    print(f"Error listing models: {e}")
