import gradio as gr
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
from swift.llm import get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
from swift.tuners import Swift
from modelscope import snapshot_download


def load_model():
    ckpt_dir = snapshot_download('andytl/ai_luo')
    model_type = ModelType.qwen_7b_chat

    template_type = get_default_template_type(model_type)
    model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
    model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    template = get_template(template_type, tokenizer)
    
    return model, tokenizer, template


def ai_luo(prompt): 
    response, history = inference(model, template, prompt)
    return response

    
model, tokenizer, template = load_model()
device = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

webui = gr.Interface(
    ai_luo, 
    inputs=[gr.Textbox(label="提出你的问题", lines=5)],
    outputs=[gr.Textbox(label="模拟罗胖口吻回答", lines=5)],
    title="AI罗胖",
    allow_flagging='never') 

webui.launch()
