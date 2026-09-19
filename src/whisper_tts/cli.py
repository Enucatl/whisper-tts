"""Command-line interface for transcript correction."""

import asyncio

import click
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from shared_inference import InferenceClient, InferenceError


@click.command()
@click.option("--input-file", type=click.File("rb"), required=True)
@click.option("--deepgram-api-key", required=True)
@click.option("--deepgram-model", default="nova-3")
@click.option("--openrouter-api-key", required=True)
@click.option("--openrouter-model", default="openai/gpt-5.6-luna")
@click.option("--language", default="it")
@click.option("--smart_format/--no_smart_format", default=True)
def main(  # noqa: PLR0913
    input_file: click.File,
    deepgram_api_key: str,
    deepgram_model: str,
    openrouter_api_key: str,
    openrouter_model: str,
    language: str,
    smart_format: bool,  # noqa: FBT001
) -> None:
    """Transcribe an audio file and interactively correct its transcript.

    Args:
        input_file: Audio file to transcribe.
        deepgram_api_key: Deepgram API key.
        deepgram_model: Deepgram transcription model.
        openrouter_api_key: OpenRouter API key.
        openrouter_model: OpenRouter model slug.
        language: Two-letter language code for the transcript.
        smart_format: Whether Deepgram should format the transcript.

    """
    source: FileSource = {"buffer": input_file}
    options = PrerecordedOptions(
        model=deepgram_model,
        language=language,
        smart_format=smart_format,
    )
    deepgram = DeepgramClient(deepgram_api_key)
    response = deepgram.listen.rest.v("1").transcribe_file(source, options)
    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    click.echo(transcript)
    if not transcript:
        error_message = "transcript is empty"
        raise ValueError(error_message)
    asyncio.run(
        _correction_session(
            transcript,
            language,
            openrouter_api_key,
            openrouter_model,
        )
    )


async def _correction_session(
    transcript: str,
    language: str,
    openrouter_api_key: str,
    openrouter_model: str,
) -> None:
    """Correct a transcript and run the interactive correction session.

    Args:
        transcript: Raw transcript returned by Deepgram.
        language: Two-letter language code for the transcript.
        openrouter_api_key: OpenRouter API key.
        openrouter_model: OpenRouter model slug.

    """
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
    messages = [{"role": "user", "content": correction_prompt}]
    client = InferenceClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        provider="openrouter",
        domain="whisper_tts",
        timeout=300,
    )
    try:
        corrected_transcript = await _openrouter_chat(
            client, openrouter_model, messages
        )
        messages.append({"role": "assistant", "content": corrected_transcript})
        click.echo("""

          Corrected transcript:


          """)
        click.echo(corrected_transcript)

        click.echo("\n" + "=" * 50)
        click.echo("INTERACTIVE CORRECTION SESSION STARTED")
        click.echo("You can now ask follow-up questions or ask for further edits.")
        click.echo("Type 'quit' or 'exit' to end the session.")
        click.echo("=" * 50 + "\n")

        while True:
            user_input = click.prompt("You", default="", show_default=False)

            if user_input.lower() in {"quit", "exit"}:
                click.echo("\nEnding interactive session. Goodbye!")
                break

            if not user_input:
                continue

            try:
                messages.append({"role": "user", "content": user_input})
                response = await _openrouter_chat(client, openrouter_model, messages)
                messages.append({"role": "assistant", "content": response})
                click.echo("\nAI Assistant:")
                click.echo(response)
                click.echo("-" * 20)
            except InferenceError as error:
                click.echo(f"An error occurred during chat interaction: {error}")
                break
    finally:
        await client.close()


async def _openrouter_chat(
    client: InferenceClient, model: str, messages: list[dict[str, str]]
) -> str:
    """Send a chat completion request to OpenRouter.

    Args:
        client: OpenRouter inference client.
        model: OpenRouter model slug.
        messages: OpenAI-compatible conversation messages.

    Returns:
        The assistant's response text.

    """
    response = await client.complete(model=model, messages=messages)
    if not response.content:
        error_message = "OpenRouter returned an empty response"
        raise ValueError(error_message)
    return response.content.strip()
