import os
import json

from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')

def read_json(file_path):
    with open(file_path, 'r') as f:
     return json.load(f)

def read_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()


query_task = (
    "The provided detection results are accurate and should be used to generate your response. Your response must focus on the following detection categories: vehicles (cars, trucks, SUVs, buses), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (e.g., no parking, warning, directional, no-entry, pedestrian crossing, no horn, speed limit, etc.), traffic lights (red, green, yellow, unclear), traffic cones, barriers (concrete, metal, temporary yellow), and miscellaneous objects (e.g., debris, dustbins, animals, booth-like structures, etc.). Do not include or infer any objects or categories not present in the detection results. \n\n"
    "If a detection category is empty or not mentioned, do not generate any content for that category. Provide descriptions only for categories with relevant detections.\n\n"
)

query_workflow = (
    "Workflow (Step by Step):\n"
    "1.Begin by analyzing the detection results across all relevant categories: vehicles, vulnerable road users, traffic signs, traffic lights, traffic cones, barriers, and other objects.\n"
    f"2.For each detected object, note its position, type, direction, and depth information (indicating its distance from the ego car). Depth will help assess the urgency and nature of the car's response.\n"
    f"3.Create a clear and structured description for each detected object. Ensure the description reflects the object's relevance based on its proximity and potential impact on the ego car.\n"
    f"4.Accompany each description with an explanation of how the object affects the ego car's driving behavior, considering both its position and distance.\n"
    f"5.Review the descriptions and explanations for all categories. Focus on how the objects interact with each other and influence the ego car's decisions. Arrange the descriptions and explanations logically, prioritizing objects that are more critical to the ego car's immediate decision-making process.\n"
)

query_format=(
    "Use the following JSON structure:\n"
    """{
  "vehicles": [{"description": "xxx", "explanation": "xxx"}],
  "vulnerable_road_users": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_signs": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_lights": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_cones": [{"description": "xxx", "explanation": "xxx"}],
  "barriers": [{"description": "xxx", "explanation": "xxx"}],
  "other_objects": [{"description": "xxx", "explanation": "xxx"}],
  "overall_scenario": "xxx"
}"""
)

query_guide = (
    "\n\nGuidelines:\n"
    "1.Identify and Describe Significant Objects:\n"
    f"- Specify each object's appearance, position, distance, type, and direction, and explain its impact on the ego car's driving behavior.\n"
    "2.Vehicle Interactions:\n"
    "- Explain how the ego car should interact with vehicles in its lane and adjacent lanes.\n"
    "- Highlight necessary precautions, such as maintaining safe distances and adjusting speed, especially with large or slow-moving vehicles like trucks.\n"
    "3.Interactions with Vulnerable Road Users:\n"
    "- Describe the locations and movements of pedestrians, cyclists, and motorcyclists."
    "- Detail how the ego car should ensure their safety, especially if they are near or might enter the travel lane.\n"
    "4.Traffic Signs and Signals:\n"
    "- Interpret the significance of visible traffic signs and lights.\m"
    "- Provide guidance on how the ego car should respond to ensure compliance with traffic rules and safe navigation. If signals are unclear or not visible, mention alternative cues the car should use.\n"
    "Environmental and Road Conditions:\n"
    f"- Discuss how weather, time of day, and road type affect the ego car's driving strategy.\n"
    "- Note any conditions, such as narrow lanes or construction zones, that may require driving adjustments.\n\n"
    "Detection Results:\n"
)

query_rules = (
    "\n\nImportant:\n"
"""1.Do not mention or infer any objects or categories not present in the detection results. If a category is empty or not listed, return an empty list for that category (e.g., "traffic_cones": []).\n"""
"""2.For "overall_scenario": Convert structured text that generated before into coherent text. The "overall_scenario" should describe each detected object's appearance, position, direction, distance and explain why it affects the ego car's behavior."""
)


query_task_1=(
    """Task (Stage 1 - Structured Description and Explanation):
Your task is to generate a structured JSON output with detailed descriptions and explanations for each detected category. The output should include:
- Detailed descriptions of each detected element (e.g., vehicles, cyclists, pedestrians, traffic signs, traffic lights, traffic cones, barriers).
- The position, movement, and importance of each element relative to the ego car.
- How each element might impact the ego car's driving decisions (e.g., speed adjustments, lane changes, stopping).

Ensure that empty categories are represented as empty arrays.

Output Structure (Stage 1):
{
  "vehicles": [{"description": "xxx", "explanation": "xxx"}],
  "vulnerable_road_users": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_signs": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_lights": [{"description": "xxx", "explanation": "xxx"}],
  "traffic_cones": [{"description": "xxx", "explanation": "xxx"}],
  "barriers": [{"description": "xxx", "explanation": "xxx"}],
  "other_objects": [{"description": "xxx", "explanation": "xxx"}],
}"""
)

query_task_2 = """Task (Stage 2 - General Perception):
Based on the structured descriptions and explanations provided in Stage 1, your task is to generate a general perception of the scene. This should include:
- A summary of the overall traffic situation and key elements affecting the ego car's driving behavior.
- Identification of the most critical elements or threats in the environment.
- An overview of how these elements collectively influence the ego car's path and decision-making process."""

query_task_3 = """Task (Stage 2 - Driving Suggestion):
Based on the structured output from Stage 1 and the general perception provided in the previous Stage 2, your task is to generate specific driving suggestions for the ego car. Your suggestions should include:
- Specific actions the ego car should take to navigate the traffic scenario safely.
- Recommendations on speed adjustments, lane changes, or stopping based on the observed elements.
- Any additional precautions the ego car should take to avoid potential hazards."""