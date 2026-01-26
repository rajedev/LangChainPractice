"""
Author: Rajendhiran Easu
Date: 26/01/26
Description: Hugging Face text and multimodal examples
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()

# Check if HuggingFace token is set
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("⚠️  Warning: HUGGINGFACEHUB_API_TOKEN not found in .env file")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("Add to .env file: HUGGINGFACEHUB_API_TOKEN=your_token_here\n")


print("="*60)
print("TEXT GENERATION EXAMPLE")
print("="*60)
try:
    text_llm = HuggingFaceEndpoint(
        repo_id="zai-org/GLM-4.7-Flash",
        temperature=0.2
    )
    chat_hf = ChatHuggingFace(llm=text_llm)

    print("🔧 Calling HuggingFace API...")
    result = chat_hf.invoke("what is hugging face in langchain in 15 words?")

    print(f"\n✅ Success!")
    print(f"Question: what is hugging face in langchain in 15 words?")
    print(f"Answer: {result}")
    print()

except Exception as e:
    print(f"\n❌ Error: {e}")




# print("=" * 60)
# print("MULTIMODAL (VISION) EXAMPLE")
# print("=" * 60)
# try:
#     vision_llm = HuggingFaceEndpoint(
#         repo_id="lightonai/LightOnOCR-2-1B",  # Vision-capable model
#         temperature=0.2
#     )
#
#     h_message = [HumanMessage(
#         content=[
#             {
#                 "type": "text",
#                 "text": "Explain what you see in this image"
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": "https://www.google.com/logos/doodles/2026/india-republic-day-2026-6753651837111184.2-2x.png"
#                 }
#             }
#         ]
#     )]
#
#     chat_llm = ChatHuggingFace(llm=vision_llm)
#     result = chat_llm.invoke(h_message)
#     print(f"Image URL: https://www.google.com/logos/doodles/2026/india-republic-day-2026-6753651837111184.2-2x.png")
#     print(f"Response: {result}")
#
# except Exception as e:
#     print(f"⚠️  Multimodal not supported with this setup: {e}")
#     print("\nNote: HuggingFace Endpoint API has limited multimodal support.")
