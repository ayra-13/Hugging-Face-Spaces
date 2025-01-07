# app.py
import gradio as gr
from model import predict

# Define the Gradio interface
def inference(number):
    result = predict(int(number))
    return f"The number {number} is {result}."

# Create the Gradio app
interface = gr.Interface(
    fn=inference, 
    inputs="number", 
    outputs="text", 
    title="Even or Odd Predictor",
    description="Enter a number to predict whether it's even or odd."
)

if __name__ == "__main__":
    interface.launch()
