import gymnasium as gym
import numpy as np
import time
import policy_2205

def watch_landing(model_path="best_policy_2205.npy", episodes=5, fps=50):
    """
    Renders the trained lunar lander agent playing in real-time.
    Prints detailed telemetry to verify how the lander controls descent,
    orientation, and touchdown.
    """
    # Load policy weights
    policy = np.load(model_path)
    
    # Initialize environment with human rendering
    env = gym.make("LunarLander-v3", render_mode="human")
    
    action_names = {
        0: "Drift (No thrust)",
        1: "Left Thruster",
        2: "Main Descent Thruster",
        3: "Right Thruster"
    }

    print("=" * 65)
    print("🚀 LUNAR LANDER LANDING VISUALIZER (Group 2205)")
    print("=" * 65)
    print("Watch the Pygame window to see the lander perform.")
    print("Press Ctrl+C in terminal or close the window to stop.\n")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        print(f"\n--- Episode {ep} Started ---")

        while not done:
            # Predict action from policy
            action = policy_2205.policy_action(policy, obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Extract telemetry from observation vector:
            # obs: [x, y, vx, vy, angle, v_angular, left_leg_contact, right_leg_contact]
            x, y, vx, vy, angle, v_ang, left_leg, right_leg = obs

            # Real-time telemetry printout (overwrites single line)
            print(
                f"\rStep: {step_count:3d} | "
                f"Pos: ({x:+.2f}, {y:+.2f}) | "
                f"Vel: ({vx:+.2f}, {vy:+.2f}) | "
                f"Tilt: {angle:+.2f} rad | "
                f"Legs: [{'L' if left_leg else ' '}{'R' if right_leg else ' '}] | "
                f"Reward: {total_reward:+.1f}",
                end="",
                flush=True
            )

        # Final outcome diagnosis
        left_leg_down = bool(obs[6])
        right_leg_down = bool(obs[7])
        y_pos = obs[1]
        x_pos = obs[0]
        
        print("\n" + "-" * 65)
        print(f"Episode {ep} Finished in {step_count} steps! Final Score: {total_reward:.2f}")

        # Verification diagnosis
        if total_reward >= 200:
            status = "🎉 PERFECT SOFT LANDING (+100 bonus awarded)"
        elif total_reward >= 100:
            status = "✅ SAFE LANDING (Low impact, upright)"
        elif left_leg_down and right_leg_down:
            status = "⚠️ TOUCHDOWN (Both legs down, minor fuel penalty)"
        else:
            status = "❌ CRASH OR OUT OF BOUNDS"

        print(f"Landing Status: {status}")
        print(f"Final Landing Position: x = {x_pos:+.2f} (Landing flags are at x ≈ ±0.2)")
        print(f"Leg Contacts: Left Leg = {'TOUCHING' if left_leg_down else 'OFF'}, Right Leg = {'TOUCHING' if right_leg_down else 'OFF'}")
        print("=" * 65)

        time.sleep(1.0)  # Brief pause between episodes

    env.close()

if __name__ == "__main__":
    watch_landing()
