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
    click.echo(transcript)
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
                click.echo(model)
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
        click.echo("""

              Corrected transcript:


              """)
        click.echo(corrected_transcript)

    # --- 3. Interactive Chat Session ---
    if google_api_key is not None:
        click.echo("\n" + "=" * 50)
        click.echo("INTERACTIVE CORRECTION SESSION STARTED")
        click.echo("You can now ask follow-up questions or ask for further edits.")
        click.echo("Type 'quit' or 'exit' to end the session.")
        click.echo("=" * 50 + "\n")

        # Initialize the chat history with the initial transcription and correction
        chat = gemini_model.start_chat(
            history=[
                {"role": "user", "parts": [correction_prompt]},
                {"role": "model", "parts": [corrected_transcript]},
            ]
        )

        while True:
            user_input = click.prompt("You", default="", show_default=False)

            if user_input.lower() in {"quit", "exit"}:
                click.echo("\nEnding interactive session. Goodbye!")
                break

            if not user_input:
                continue

            try:
                # Send the user's message to the ongoing chat session
                response = chat.send_message(user_input)

                # Print the model's response (which maintains context)
                click.echo("\nAI Assistant:")
                click.echo(response.text.strip())
                click.echo("-" * 20)

            except Exception as e:
                click.echo(f"An error occurred during chat interaction: {e}")
                break
    else:
        click.echo(
            "\nSkipping interactive chat because GOOGLE_API_KEY was not provided."
        )


if __name__ == "__main__":
    _main()
