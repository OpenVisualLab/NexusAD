

def gen_query_description(image_info):
    base_description = (
        "Now you are the autonomous driving system of a vehicle (ego car) analyzing the provided image in a traffic scenario"
    )

    optional_info = []
    if "period" in image_info:
        optional_info.append(f"during \"{image_info['period']}\"")
    if "weather" in image_info:
        optional_info.append(f"in \"{image_info['weather']}\" weather")
    if "city" in image_info:
        optional_info.append(f"located in the city of \"{image_info['city']}\"")
    if "location" in image_info:
        optional_info.append(f"on a \"{image_info['location']}\"")

    if optional_info:
        optional_description = ", ".join(optional_info)
        full_description = f"{base_description} {optional_description}."
    else:
        full_description = base_description + "."

    return full_description


query_objects = ("The following detection results represent the objects and their positions within the scene. These detections are fully accurate and should be used as the basis for generating your answers. Do not mention or infer any objects or categories that are not present in the detection results. If a detection category is empty or not mentioned in the detection results, do not generate any description or explanation for that category. Only provide descriptions for categories that contain relevant detections.\n\n Detection results:\n")

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

