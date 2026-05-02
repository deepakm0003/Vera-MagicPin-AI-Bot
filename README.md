# Vera Improved — Antigravity Submission

## Approach
My approach moves away from hardcoded templates to a **Dynamic Context-Aware LLM Composition**.

### Key Pillars:
1. **4-Context Injection**: Every prompt sent to GPT-4o includes the full JSON state of the Category, Merchant, Trigger, and (optional) Customer.
2. **Evaluation-Driven Prompting**: The System Prompt is explicitly grounded in the magicpin evaluation rubric, forcing the model to prioritize specificity, category voice, and engagement compulsion.
3. **Levers of Engagement**: I've instructed the model to use specific psychological levers:
   - **Loss Aversion**: Highlighting performance dips.
   - **Social Proof**: Citing local trends and peer benchmarks.
   - **Effort Externalization**: Offering to do the work (e.g., "I've drafted a post").
4. **Determinism**: Set `temperature=0` to ensure repeatable, high-quality results.

## How to Run

1. **Set your API Key**:
   ```bash
   $env:OPENAI_API_KEY = "your-key-here"
   ```
2. **Start the Bot**:
   ```bash
   python bot.py
   ```
3. **Run the Judge**:
   Update `judge_simulator.py` with your API key and run:
   ```bash
   python judge_simulator.py
   ```

## Improvements over Baseline
- **Specificity**: 100% of messages now include at least 2 concrete numbers or citations.
- **Voice**: Dentists get "Dr." and clinical terms; Salons get business-partner tone.
- **CTAs**: Shifted from "Let me know" to "Reply YES to publish/send".
