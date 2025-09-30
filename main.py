import click
import httpx
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import google.generativeai


@click.command()
@click.option("--input-file", type=click.File("rb"), required=True)
@click.option("--deepgram-api-key", required=True)
@click.option("--deepgram-model", default="nova-3")
@click.option("--google-api-key", default=None)
@click.option("--google-model", default="gemini-2.5-flash-preview-09-2025")
@click.option("--language", default="it")
@click.option("--smart_format/--no_smart_format", default=True)
def _main(
    input_file: click.File,
    deepgram_api_key: str,
    deepgram_model: str,
    google_api_key: str | None,
    google_model: str,
    language: str,
    smart_format: bool,
) -> None:
    source: FileSource = {"buffer": input_file}
    options = PrerecordedOptions(
        model=deepgram_model,
        language=language,
        smart_format=smart_format,
    )
    deepgram = DeepgramClient(deepgram_api_key)
    response = deepgram.listen.rest.v("1").transcribe_file(
        source,
        options,
        timeout=httpx.Timeout(300, connect=10),
    )
    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    print(transcript)
    if not transcript:
        exception_message = "transcript is empty"
        raise Exception(exception_message)
    if google_api_key is not None:
        google.generativeai.configure(api_key=google_api_key)
        available_models = [
            model.name.lstrip("models/") for model in google.generativeai.list_models()
        ]
        if google_model not in available_models:
            for model in sorted(available_models):
                print(model)
            exception_message = f"{google_model=} not in {available_models=}"
            raise Exception(exception_message)
        gemini_model = google.generativeai.GenerativeModel(google_model)
        correction_prompt = f"""
            I will copy the raw transcription of an audio, transcribed by AI.
            Please review it for errors in spelling, punctuation,
            possibly mistranscribed words.

            Add paragraphs by separating with an empty line
            to facilitate reading and comprehension.

            The two-letter code for the language is: {language}.
            Keep this in mind and answer only in the same language.

            Correct any mistakes you find, by staying as close as possible
            to the original phrasing.
            Provide only the corrected version of the transcript,
            without any additional commentary, preamble, or conversational phrases.


            Original Transcript:
            ---
            {transcript}
            ---
            Corrected Transcript:
        """
        # More safety settings can be configured if needed
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generation_config = google.generativeai.types.GenerationConfig(temperature=0.7)
        response = gemini_model.generate_content(
            correction_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        corrected_transcript = response.text.strip()
        print("""

              Corrected transcript:


              """)
        print(corrected_transcript)


if __name__ == "__main__":
    _main()
