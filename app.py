import cv2
import gradio as gr
import spaces
from ultralytics import YOLO, solutions

css = '''
.gradio-container{max-width: 600px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''
js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
# Load YOLO models
yolo_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt.pt")
names = yolo_model.model.names

# Initialize Solutions
speed_obj = solutions.SpeedEstimator(
    reg_pts=[(0, 360), (1280, 360)],
    names=names,
    view_img=False,
)

region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=region_points,
    classes_names=yolo_model.names,
    draw_tracks=True,
    line_thickness=2,
)

dist_obj = solutions.DistanceCalculation(names=names, view_img=False)

gym_object = solutions.AIGym(
    line_thickness=2,
    view_img=False,
    pose_type="pushup",
    kpts_to_check=[6, 8, 10],
)

heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=False,
    shape="circle",
    classes_names=yolo_model.names,
)

def process_video(video_path, function):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    output_path = f"{function}_output.avi"
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        
        if function == "Speed Estimation":
            tracks = yolo_model.track(im0, persist=True, show=False)
            im0 = speed_obj.estimate_speed(im0, tracks)
        elif function == "Object Counting":
            tracks = yolo_model.track(im0, persist=True, show=False)
            im0 = counter.start_counting(im0, tracks)
        elif function == "Distance Calculation":
            tracks = yolo_model.track(im0, persist=True, show=False)
            im0 = dist_obj.start_process(im0, tracks)
        elif function == "Workout Monitoring":
            results = pose_model.track(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results)
        elif function == "Heatmaps":
            tracks = yolo_model.track(im0, persist=True, show=False)
            im0 = heatmap_obj.generate_heatmap(im0, tracks)

        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    return output_path

def gradio_interface(video, function):
    processed_video_path = process_video(video, function)
    return processed_video_path

# Create Gradio Blocks Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme", js=js_func) as demo:
    gr.Markdown("# YOLOv8 Ultralytics Solutions")
    
    with gr.Row():
        video_input = gr.Video()
        function_selector = gr.Dropdown(
            choices=["Speed Estimation", "Object Counting", "Distance Calculation", "Workout Monitoring", "Heatmaps"],
            label="Select Function",
            value="Speed Estimation",
        )
    
    output_video = gr.Video()
    process_button = gr.Button("Process Video")
    
    process_button.click(fn=gradio_interface, inputs=[video_input, function_selector], outputs=output_video)

    gr.Examples(
        examples=[
            ["assets/SpeedEstimation.mp4", "Speed Estimation"],
            ["assets/ObjectCounting.mp4", "Object Counting"],
            ["assets/WorkoutMonitoring.mp4", "Workout Monitoring"],
            ["assets/Heatmaps.mp4", "Heatmaps"],
            ["assets/DistanceCalculation.mp4", "Distance Calculation"],
        ],
        inputs=[video_input, function_selector]
    )

    gr.Markdown("⚠️ The videos that are 30 seconds or less will avoid time consumption issues and provide better accuracy.")

demo.launch(share=True)
