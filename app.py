from tempfile import TemporaryDirectory
from typing import List

from pytube import YouTube
import whisper
from transformers import pipeline
import gradio as gr


def get_title(url: str) -> str:
    """Returns the title of the YouTube video at the given URL."""
    yt = YouTube(url)
    return f"**{yt.title}**"


def download_audio(url: str, path: str) -> None:
    """Downloads the audio from the YouTube video at the given URL and saves it to the specified path."""
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    audio.download(output_path=path, filename="a.mp4")


def transcribe(path: str) -> List[str]:
    """Transcribes the audio file at the given path and returns the text."""
    model = whisper.load_model("tiny")
    transcription = model.transcribe(path, fp16=False)["text"]
    transcription_chunks = [transcription[i : i + 1000] for i in range(0, len(transcription), 1000)]
    return transcription_chunks


def summarize(transcription: List[str]) -> str:
    """Summarizes the given text and returns the summary."""
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_chunks = model(transcription, max_length=80, min_length=30)
    summary = (" ".join([chunks["summary_text"] for chunks in summary_chunks]).strip().replace(" . ", ". "))
    return summary


def execute_pipeline(url: str) -> str:
    """Generates a temporary directory and executes the pipeline to download, transcribe and summarize the video."""
    with TemporaryDirectory(dir=".") as tmp_dir:
        download_audio(url, tmp_dir)
        result = transcribe(f"{tmp_dir}/a.mp4")
        text = summarize(result)
        return text


def main() -> None:
    """Generates the Gradio interface."""
    with gr.Blocks(analytics_enabled=True, title="Summarize a video") as page:
        gr.HTML('<h2 style="text-align:center"><span style="font-size:36px">Summarize a <strong>Youtube</strong> video</span></h2>')
        url = gr.Textbox(label="Enter the URL:")
        title = gr.Markdown()
        output = gr.Textbox(label="Summary")
        summarize_btn = gr.Button("Go!").style(full_width=False)
        summarize_btn.click(fn=get_title, inputs=url, outputs=title)
        summarize_btn.click(fn=execute_pipeline, inputs=url, outputs=output)
        gr.Markdown("*Works best with videos under 10 minutes. It usually takes around 2-3 minutes to execute.*")
    page.launch()


if __name__ == "__main__":
    main()