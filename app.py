#!/usr/bin/env python3

"""
Hiroyuki-SLM API with Hugging Face Spaces
"""

import asyncio
import gradio as gr
from slm_model import HiroyukiSLM

slm = HiroyukiSLM()

def respond(text):
    return asyncio.run(slm.generate(text))

gr.Interface(fn=respond, inputs="text", outputs="text").launch()
