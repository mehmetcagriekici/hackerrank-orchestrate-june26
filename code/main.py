import sys
from agent.agent import Agent

def main():
    """
    Main entry point. Run Agent on full dataset.
    """
    try:
        agent = Agent(
            ollama_url="http://localhost:11434",
            model_name="llama3.2:1b"
        )
        
        success = agent.run(
            input_csv='dataset/claims.csv',
            output_csv='output.csv'
        )
        
        if success:
            print("\nComplete. Output written to output.csv")
            return 0
        else:
            print("\n? Failed. Check log file for details.")
            return 1
    
    except Exception as e:
        print(f"\n? Fatal error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
