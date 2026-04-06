from env import MigrationEnv, MigrationAction

def main():
    print("🚀 Initializing Migration Environment...")
    env = MigrationEnv()
    
    # 1. Start the episode
    obs = env.reset()
    
    print("\n" + "="*50)
    print("🎯 WHAT THE LLM SEES (The State/Prompt):")
    print("="*50)
    print(f"Status: {obs.status}")
    print(f"Message: {obs.message}")
    print("\n--- CONTEXT ---")
    print(obs.context)
    print("="*50)

    # 2. Mock the Agent's Action 
    # The JSON says the first task is 'quick_ping' in 'test2\test.py'
    # We will pretend the LLM generated this fixed code:
    mock_fixed_code = """import httpx as rq

def quick_ping(url):
    \"\"\"Uses httpx now!\"\"\"
    return rq.get(url).status_code
"""
    
    print("\n🤖 AGENT SUBMITTING CODE...")
    
    # Notice we use the file_path provided by the current task
    action = MigrationAction(
        file_path=env.current_task["file_name"], 
        new_code=mock_fixed_code
    )
    
    # 3. Take the step (Write file + Run Grader)
    obs, reward, done, info = env.step(action)
    
    print("\n" + "="*50)
    print("🏆 ENVIRONMENT REWARD / RESULT:")
    print("="*50)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Status: {obs.status}")
    print(f"Message: {obs.message}")
    if obs.context:
        print(f"Grader Output:\n{obs.context}")

if __name__ == "__main__":
    main()