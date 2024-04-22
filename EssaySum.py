import gradio as gr
import requests

API_URL = "https://api-inference.huggingface.co/models/WilliamHo/bart-finetuned-model"
headers = {"Authorization": "Bearer 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"}
MAX_CHUNK_SIZE = 4000

def summarize_chunk(chunk, length_penalty, num_beams, max_length, repetition_penalty):
    gen_kwargs = {
        'length_penalty': length_penalty,
        'num_beams': num_beams,
        'max_length': max_length,
        'repetition_penalty': repetition_penalty
    }
    payload = {
        'inputs': chunk,
        'parameters': gen_kwargs
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()
        generated_text = output[0]['generated_text']
        # a simple formatting to the summary
        filtered_text = generated_text.replace("generated_text:", "").replace("background", "**background**").replace("purpose", "**purpose**").replace("results", "**results**").replace("conclusions", "**conclusions**").replace("abstract", "**abstract**").replace("objective","**objective**")
        return filtered_text
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def query(text, length_penalty, num_beams, max_length, repetition_penalty):
    chunks = [text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
    summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk, length_penalty, num_beams, max_length, repetition_penalty)
        summaries.append(summary)
    joined_summary = " ".join(summaries)
    if len(joined_summary) <= MAX_CHUNK_SIZE:
        final_summary = summarize_chunk(joined_summary, length_penalty, num_beams, max_length, repetition_penalty)
        return final_summary
    else:
        return query(joined_summary, length_penalty, num_beams, max_length, repetition_penalty)

summarization = gr.Interface(
    fn=query,
    inputs=[
        gr.Textbox(label="Text"),
        gr.Slider(label="Length Penalty", 
                  minimum=0.1, maximum=3, value=1, step=0.1),
        gr.Slider(label="Num Beams", 
                  minimum=1, maximum=20, value=8, step=1),
        gr.Slider(label="Max Length", 
                  minimum=50, maximum=500, value=200, step=10),
        gr.Slider(label="Repetition Penalty", 
                  minimum=1, maximum=2, value=1, step=0.1)
    ],
    outputs=["text"],
)

if __name__ == "__main__":
    summarization.launch(share=True)