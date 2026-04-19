from setup_qa import create_clients, get_qa


def main():
    print("Initializing system...")

    pc, embedding = create_clients()
    qa = get_qa(pc, embedding)

    print("Ready! Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() == "exit":
                print("Exiting...")
                break

            if not query:
                print("Enter something.")
                continue

            print("Thinking...")

            result = qa.invoke({"input": query})
            answer = result.get("answer", result)

            print("Bot:", answer)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
