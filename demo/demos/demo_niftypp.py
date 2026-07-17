import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from demos.resources.theme import *
import demos.backend as backend
import algorithms.algorithm_PCAGMM_latent as algo
from demos.state_type import AppState

# TODO
DISPLAY_DEVICE_CHOICE = True
ui_device = "GPU"

def update_state(value, field, state):
    setattr(state, field, value)
    return state

def process_run(state:AppState):
    for im, updated_state in backend.process_run(state):
        yield im, updated_state


def resample_initialisation(state:AppState):
    return backend.process_init(state)

def resample_initialisation_noised(state:AppState):
    return backend.process_algo_init(state)

def process_advanced_display(state:AppState):
    return backend.process_advanced_display(state)

def update_algorithm_dataset(state:AppState):
    return backend.update_algorithm_dataset(state)

def demo_niftypp():
    with gr.Blocks(fill_height=True, fill_width=True) as demo_niftypp : 
        
        default_state = AppState()
        app_state = gr.State(default_state) # instantiated default state
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, visible=DISPLAY_DEVICE_CHOICE):
                    in_processing_unit_choice = gr.Radio(
                        label="Computation device",
                        choices=["GPU", "CPU"],
                        value=ui_device,
                        type="value",
                        elem_classes="radio_group",
                        elem_id="processing_unit_radio_group"
                    )
                    
        gr.HTML(HTML_H_SEPARATOR)
        
        with gr.Accordion(label="Dataset Sources", open=False):
            gr.Markdown("Check out the [github repository](https://github.com/MarageDev/RMGD) to see how to load and generate a custom PCAGMM Generator to test the algorithm on a custom dataset")
            with gr.Row(equal_height=True, variant="default"):
                file_algorithm_dataset = gr.File(label="Algorithm Dataset", value=default_state.batch_tensor_path, interactive=True)
                file_initialisation_pcagmm = gr.File(label="Initialisation Model", value=default_state.pcagmm_model_path)
            
        gr.HTML(HTML_H_SEPARATOR)       
            
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### Main Settings")
                sldr_algorithm_samples = gr.Slider(
                    label="Algorithm Sample Amount",
                    info="Number of image samples to use for the algorithm (high = important computation time)",
                    minimum=1,
                    maximum=2,
                    value=default_state.algo_samples,
                    step=1,
                    precision=0,
                    interactive=True
                )
                sldr_algorithm_samples.change(
                    fn=lambda v, s: update_state(v, "algo_samples", s),
                    inputs=[sldr_algorithm_samples, app_state],
                    outputs=app_state
                )
                
                with gr.Row(equal_height=True):
                    sldr_patch_size = gr.Slider( # Patch size power (2^x) with x patch size to avoid beige corner border
                        label="Patch Size",
                        info="Size of the patches to retrieve from the dataset",
                        minimum=1,
                        maximum=6,
                        value=default_state.patch_size_pow,
                        step=1,
                        precision=0,
                        interactive= True
                    )
                    _update_patch_size = sldr_patch_size.change(
                        fn=lambda v, s: update_state(v, "patch_size_pow", s),
                        inputs=[sldr_patch_size, app_state],
                        outputs=app_state
                    ).then(
                        fn=lambda v, s: update_state(2**v, "patch_size", s),
                        inputs=[sldr_patch_size, app_state],
                        outputs=app_state
                    )
                    
                    sldr_patch_stride = gr.Slider(
                        label="Stride",
                        info="Amount of skipped pixels when searching for patches (high = lower computation time)",
                        minimum=1,
                        maximum=default_state.patch_size,
                        value=default_state.stride,
                        step=1,
                        precision=0,
                        interactive= True
                    )
                    sldr_patch_stride.change(
                        fn=lambda v, s: update_state(v, "stride", s),
                        inputs=[sldr_patch_stride, app_state],
                        outputs=app_state
                    )
                    _update_patch_size.then(
                        fn = lambda x, y : gr.update(maximum = max(x,2), value= min(x,y)),
                        inputs=[sldr_patch_size, sldr_patch_stride],
                        outputs=sldr_patch_stride
                    )
                
                sldr_iterations = gr.Slider(
                    label="Iterations",
                    info="Number of iterations used to solve the flow matching ODE (number of timesteps between 0 and 1)",
                    minimum=1,
                    maximum=100,
                    value=default_state.iterations,
                    step=1,
                    precision=0,
                    interactive= True
                )
                sldr_iterations.change(
                    fn=lambda v, s: update_state(v, "iterations", s),
                    inputs=[sldr_iterations, app_state],
                    outputs=app_state
                )
                
                with gr.Row(equal_height=True):
                    drp_weight_type = gr.Dropdown(
                        label="Weight Type", 
                        info="Type of weight applied to the mask for patch aggregation", 
                        choices=["gaussian", "default"], 
                        value=default_state.weight_type, 
                        interactive=True
                    )
                    drp_weight_type.change(
                        fn=lambda v, s: update_state(v, "weight_type", s),
                        inputs=[drp_weight_type, app_state],
                        outputs=app_state
                    )
                
                    drp_scheduler = gr.Dropdown(
                        label="Scheduler", 
                        info="Scheduler used for the timesteps repartition",
                        choices=["linear", "cosine", "quad"], 
                        value=default_state.scheduler, 
                        interactive=True
                    )
                    drp_scheduler.change(
                        fn=lambda v, s: update_state(v, "scheduler", s),
                        inputs=[drp_scheduler, app_state],
                        outputs=app_state
                    )
            
            
                with gr.Row(equal_height=True, variant="default"):
                    chk_use_seed = gr.Checkbox(
                        value=default_state.use_seed, 
                        label="Use Custom Seed", 
                        info="Allow to use a custom seed for reproducibility", 
                        interactive=True
                    )
                    sldr_seed = gr.Slider(
                        label="Seed",
                        info="Seed used in the algorithm",
                        minimum=0,
                        maximum=10000,
                        value=default_state.seed,
                        step=1,
                        precision=0,
                        interactive= default_state.use_seed
                    )

                    chk_use_seed.change(
                        fn = lambda x, y : gr.update(y, interactive=x), 
                        inputs= [chk_use_seed, sldr_algorithm_samples], 
                        outputs=sldr_seed, 
                        show_progress=False
                    ).then(
                        fn=lambda v, s: update_state(v, "use_seed", s), 
                        inputs=[chk_use_seed, app_state],
                        outputs=app_state
                    )
                    sldr_seed.change(
                        fn=lambda v, s: update_state(v, "seed", s),
                        inputs=[sldr_seed, app_state],
                        outputs=app_state
                    )
                btn_run = gr.Button("Run",variant="primary", size="lg", interactive=True)
                
            
            with gr.Column():
                img_generation_result   = gr.Image(label="", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image", interactive=False)
        
            run_generation = btn_run.click(
                fn = process_run,
                inputs=[app_state],
                outputs=[img_generation_result, app_state],
                queue=True
            )
        
        gr.HTML(HTML_H_SEPARATOR)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Initialisation")
                
                sldr_init_noisefact = gr.Slider(
                    label="Initial Noise Factor", 
                    info="Amount of noise to blend the sampled image with (1 = only noise, 0 = only sampled image)",
                    minimum=0,
                    maximum=1,
                    value=default_state.init_noisefact,
                    step=0.01,
                )
                sldr_init_noisefact.change(
                    fn=lambda v, s: update_state(v, "init_noisefact", s),
                    inputs=[sldr_init_noisefact, app_state],
                    outputs=app_state
                )
                
                img_sampled_init        = gr.Image(label="Sampled Initialisation", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)
                img_init                = gr.Image(label="Algorithm Initialisation", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)
                btn_sample_init = gr.Button(value="Resample Initialisation", variant="secondary", interactive=True)
                btn_sample_init.click(
                    fn = lambda : gr.update(interactive= False),
                    outputs=btn_sample_init    
                ).then(
                    fn = resample_initialisation,
                    inputs=[app_state],
                    outputs=[img_sampled_init, app_state],
                    show_progress=False
                ).then(
                    fn = lambda : gr.update(interactive= True),
                    outputs=btn_sample_init    
                ).then(
                    fn = resample_initialisation_noised,
                    inputs = [app_state],
                    outputs=[img_init, app_state],
                    show_progress=False
                )

            with gr.Column(scale = 3, elem_classes="border_left_panel",variant="panel"):
                gr.Markdown("### Advanced Display")
            
                with gr.Row(equal_height=True):
                    with gr.Column():
                        img_patch_regions       = gr.Image(label="Patch Regions", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250 nearest_upscale", interactive=False)
                        img_novelty_map         = gr.Image(label="Novelty Map", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250 nearest_upscale", interactive=False)
                
                    with gr.Column(): 
                        img_predominant_mosaic  = gr.Image(label="Predominant Mosaic", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)
                        img_predominant         = gr.Image(label="Predominant", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)
                        
                    with gr.Column(): 
                        img_coi_mosaic          = gr.Image(label="COI Mosaic", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)
                        img_coi                 = gr.Image(label="COI", type="numpy", image_mode="RGB", sources=None, buttons=["download","share","fullscreen"], container=True, elem_classes="output_image full_size_image h250", interactive=False)

        file_algorithm_dataset.change(
            fn=lambda v, s: update_state(v, "batch_tensor_path", s), 
            inputs=[file_algorithm_dataset, app_state],
            outputs=app_state
        ).then(
            fn = update_algorithm_dataset,
            inputs = app_state,
            outputs= sldr_algorithm_samples,
            show_progress=False
        )
              
        run_generation.success(
            fn = process_advanced_display,
            inputs=app_state,
            outputs=[img_patch_regions, img_novelty_map, img_predominant_mosaic, img_predominant, img_coi_mosaic, img_coi],
            queue=True
        )

"""with gr.Blocks() as pcagmm_page:
    with gr.Row(equal_height=True):
        gr.Dropdown(choices=["Celeba-HQ Encoded 512x512 (28k)", "Celeba-HQ Encoded 256x256 (28k)"], value="Celeba-HQ Encoded 512x512 (28k)", interactive=False) # TODO 
"""

if __name__ == "__main__":
    with gr.Blocks() as demo : 
        demo_niftypp()
    demo.queue().launch(css_paths="demos/resources/style.css", theme=theme)