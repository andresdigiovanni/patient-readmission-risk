from dotenv import load_dotenv

from src.ml.pipelines import TrainingPipeline


def main():
    load_dotenv()

    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
