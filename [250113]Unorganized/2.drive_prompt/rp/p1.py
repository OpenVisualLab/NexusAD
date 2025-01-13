import json

rp_prompt = """
Given an image of a traffic scene, identify and describe the object inside the red rectangular region. Focus on providing a detailed description of the object, its type, function, and how it affects the ego vehicle's driving decisions.

The output must include:
- 'region perception': A detailed description of the object, including its characteristics, role, and potential impact on driving behavior (e.g., requiring speed adjustment, lane change, yielding, etc.).
- 'category_name': A category label that best describes the object. The category should be as specific as possible, selected from the following general groups:
  - 'vehicles' (e.g., cars, trucks, buses)
  - 'vulnerable road users' (e.g., pedestrians, cyclists, motorcyclists)
  - 'traffic signs' (e.g., no parking, warning, directional)
  - 'traffic lights' (e.g., red, green, yellow)
  - 'traffic cones'
  - 'barriers' (e.g., physical barriers, road dividers)
  - 'miscellaneous' (e.g., debris, dustbin, animals)

### Guidelines for Object Classification:

- **Vehicles**:
  - Includes cars, trucks, buses, motorcycles, and other motorized vehicles.
  - Typically on the road or designated vehicle lanes, often with distinctive features like wheels, windows, and lights.
  - Their movement and presence affect driving decisions such as maintaining a safe following distance, lane keeping, and speed adjustments.
  - Vehicles can vary in size (e.g., compact cars vs. trucks) and require different responses depending on the context (e.g., yielding to a bus vs. overtaking a slow-moving car).

- **Vulnerable Road Users**:
  - Includes pedestrians, cyclists, and motorcyclists.
  - Generally found on sidewalks, pedestrian crossings, or sharing the road with vehicles.
  - Vulnerable road users require the ego vehicle to slow down, stop, or yield to ensure their safety.
  - Pedestrians may cross the road unpredictably, cyclists often use bike lanes or the road, and motorcyclists may move between vehicle lanes. 

- **Traffic Signs**:
  - Typically mounted on poles or placed alongside roads.
  - Shapes can be rectangular, triangular, or circular, and they often use reflective colors like red, blue, yellow, or white.
  - Signs convey important information to the driver, such as speed limits, warnings, and directions.
  - Differentiate traffic signs from advertisements or decorative signs, which do not provide information relevant to driving decisions.

- **Traffic Lights**:
  - Positioned at intersections or pedestrian crossings.
  - Colors include red, green, and yellow, indicating stop, go, and caution respectively.
  - Traffic lights directly control the movement of the ego vehicle by signaling when to stop, proceed, or prepare to stop.
  - Ensure to distinguish traffic lights from other similar light sources, such as decorative lights or streetlights.

- **Traffic Cones**:
  - Small, lightweight, conical objects, often bright orange with reflective stripes.
  - Used to indicate restricted areas, lane changes, or construction zones.
  - Typically temporary, requiring the ego vehicle to adjust speed or change lanes to avoid the indicated area.

- **Barriers**:
  - Includes physical barriers, road dividers, guardrails, and bollards.
  - Typically larger, heavy, and fixed in place, intended for blocking or directing traffic more permanently.
  - Barriers can separate lanes, protect pedestrians, or mark areas that are off-limits to vehicles.
  - The ego vehicle must avoid contact with these barriers and follow the road as directed by them.

- **Miscellaneous**:
  - Includes unexpected obstacles like debris, dustbins, animals, or other items that may be present on or near the road.
  - Miscellaneous items have irregular shapes and no standardized appearance, making them unpredictable.
  - The ego vehicle must navigate around these objects to avoid collision while ensuring safety.

### Important Instructions:
- Ensure that the 'category_name' aligns accurately with the 'region perception'.
- The description must clearly convey how the object influences the ego vehicle's driving decisions, considering factors like safety, road positioning, and potential obstacles.

The response format should be:
{
    'region perception': '<detailed description of the object and its impact on driving>',
    'category_name': '<specific object category>'
}
"""

# File paths
tasks = ['mini', 'train', 'test', 'val']
for task in tasks:
    input_file_path = f"/data2/datasets/drivelm/data/qas/new/new_{task}_region_perception.jsonl"
    output_file_path = f"/data2/datasets/drivelm/data/qas/icl/icl_{task}_region_perception.jsonl"

    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            data['query'] = rp_prompt.strip() 
            outfile.write(json.dumps(data) + '\n')

    print(f"Updated queries have been saved to {output_file_path}")