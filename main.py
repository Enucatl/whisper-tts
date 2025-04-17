import click
import httpx
from deepgram import DeepgramClient, PrerecordedOptions, FileSource


@click.command()
@click.option("--input_file", type=click.File("rb"), required=True)
@click.option("--api_key", required=True)
@click.option("--model", default="nova-2")
@click.option("--language", default="it")
@click.option("--smart_format/--no_smart_format", default=True)
def _main(
    input_file: click.File, api_key: str, model: str, language: str, smart_format: bool
) -> None:
    source: FileSource = {"buffer": input_file}
    options = PrerecordedOptions(
        model=model,
        language=language,
        smart_format=smart_format,
    )
    deepgram = DeepgramClient(api_key)
    response = deepgram.listen.rest.v("1").transcribe_file(
        source,
        options,
        timeout=httpx.Timeout(300, connect=10),
    )
    # print(response.to_json(indent=2))
    print(response["results"]["channels"][0]["alternatives"][0]["transcript"])


if __name__ == "__main__":
    _main()
