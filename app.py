import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(img, question):
    encoding = processor(img, question, return_tensors="pt")
    outputs = model(**encoding)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Ask a question about the image...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🤖 Multimodal AI: Visual Question Answering",
    description="Upload an image and ask any question about it. (Model: dandelin/vilt-b32-finetuned-vqa)"
)

iface.launch()
