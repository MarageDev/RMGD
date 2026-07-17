# code from Mahé DUVAL
import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from demos.resources.theme import *

import demos.demo_nifty as dnfty
from demos.demo_nifty import demo_nifty
from demos.demo_niftypp import demo_niftypp
dnfty.runs_on_server = False

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    
    nifty_header = gr.HTML(HTML_HEADER_NIFTY, visible=True)
    niftypp_header = gr.HTML(HTML_HEADER_NIFTYPP, visible=False)
    
    gr.HTML(HTML_AUTHORS)
    with gr.Tabs(selected="a"):
        with gr.Tab("Nifty", id="a") as tab_nifty:
            demo_nifty()
            
            
        with gr.Tab("Nifty++", id="b") as tab_niftypp:
            demo_niftypp()
    
    tab_nifty.select(fn = lambda : (
        gr.update(visible=True),
        gr.update(visible=False)
        ),
        outputs=[nifty_header, niftypp_header],
        show_progress=False
    )
    tab_niftypp.select(fn = lambda : (
        gr.update(visible=False),
        gr.update(visible=True),
        ),
        outputs=[nifty_header, niftypp_header],
        show_progress=False
    )
    gr.HTML(HTML_FOOTER)

demo.queue().launch(css_paths="./demos/resources/style.css",head=HTML_CUSTOM_HEAD, theme=theme)