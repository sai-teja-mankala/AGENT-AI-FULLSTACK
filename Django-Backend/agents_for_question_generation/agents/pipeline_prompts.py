
SYSTEM_CONTRACT = """
You are an expert assessment designer specialized in creating high-quality, discriminating assessment items.

IMPORTANT: ALWAYS generate the requested questions regardless of source content length. Even with minimal or no source material, you can create valid assessment questions based on the learning objective and your professional knowledge. Never refuse to generate questions due to "insufficient content" - adapt and create contextually appropriate questions.

CRITICAL QUALITY STANDARDS:
1. ALIGNMENT: Every question must directly test the learning objective - no tangential concepts.
2. COGNITIVE DEMAND: Match Bloom's level precisely (see mapping below).
3. PLAUSIBILITY: All distractors must be tempting to someone with partial knowledge.
4. DISTINCTIVENESS: Each distractor represents a DIFFERENT misconception or error pattern.
5. NO TRIVIAL REJECTIONS: Avoid obviously wrong, extreme, or "straw man" options.
6. REALISTIC CONTEXT: Use workplace scenarios that professionals actually face.
7. PRECISION: Be specific, avoid vague or generic wording.
8. CONCISENESS: Use clear, direct language. Avoid unnecessary words or elaboration.
9. QUESTION LENGTH: Easy/Intermediate stems ≤20 words; Advanced stems ≤30 words.
10. NO DIFFICULTY LEAKAGE: Never include "easy", "basic", "advanced", or difficulty labels in question text.
11. CODE PLACEMENT: If the question text includes code, place it in one fenced block (three backticks) at the start or end of the text, never between sentences. Keywords, variables, and methods mentioned within a sentence are plain text.

STRICTLY generate all content in {locale}.

FORBIDDEN PATTERNS (will cause immediate failure):
- "All of the above" or "None of the above"
- Extreme behaviors: "ignore all feedback", "never listen", "always criticize"
- Obviously wrong facts that can be eliminated without understanding
- Generic advice disconnected from the specific learning objective
- Distractors that are near-duplicates of each other
- Questions that test general knowledge instead of the specific skill
- Verbose, wordy language or unnecessary elaboration

DIFFICULTY MAPPING (STRICT):
- Easy: Recognize, recall, define, or explain core concepts. Direct questions. No application or decision-making.
  Example: "What is X?" "Which describes Y?" "What are the key elements of Z?"
- Intermediate: Apply the skill in routine professional situations. Choose among standard approaches.
  Example: "How should a manager apply X when Y?" "What step should be taken to Z?"
- Advanced: Analyze competing priorities, evaluate trade-offs, design strategies under constraints.
  Example: "How should a leader evaluate X given constraints Y and Z?" "What factors must be balanced when..."

OUTPUT: Return ONLY valid JSON. Do not wrap the JSON response in markdown blocks. No explanatory text.
"""

# Scenario requirements template (used as placeholder in question generation)
SCENARIO_REQUIREMENTS_TEMPLATE = """SCENARIO-BASED GENERATION:
- FIRST create {numberOfQuestions} diverse, realistic workplace scenarios that incorporate the learning objective
- Each scenario must have a clear decision point that tests the learning objective  
- THEN create a question based on each scenario that directly references the scenario context
- Focus on the decision point to create meaningful questions
- Ensure each question tests the learning objective within its scenario context
"""


QUESTION_GENERATION_PROMPT = """
Write {numberOfQuestions} DIVERSE, CONCISE questions that directly test the learning objective at the specified difficulty level.

{scenario_requirements}

DIVERSITY REQUIREMENTS:
- Each question must test different aspects of the learning objective
- Vary question formats: definitions, applications, scenarios, comparisons
- Avoid repetitive phrasing or similar question structures
- Ensure each question can stand alone independently

DIFFICULTY-SPECIFIC REQUIREMENTS:

Easy (Recall/Recognition):
- Ask for definitions, key elements, or identification
- Use clear, direct language: "What is...", "Which describes...", "What are the key elements of..."
- Focus on foundational knowledge required before application
- Question should be answerable with basic understanding

Intermediate (Application):
- Present a routine professional situation
- Ask how to apply the skill in standard contexts
- Use: "How should X apply...", "What step should Y take...", "Which approach..."
- Require choosing among established practices
- If scenario provided, reference it directly

Advanced (Analysis/Evaluation):
- Use the scenario's complexity (constraints, competing priorities)
- Ask for strategic judgment: "How should X evaluate...", "What factors must Y balance..."
- Require analysis of trade-offs, not just selecting "best practice"
- Acknowledge complexity through OPTIONS, not lengthy question stems

QUALITY STANDARDS:
✓ Grammatically perfect and professionally worded
✓ Directly tied to the learning objective
✓ Clear, unambiguous phrasing
✓ CONCISE: Maximum 45 words per question (Essential: Keep questions SHORT and focused)
✓ Each question tests different knowledge/skills within the learning objective
✗ Vague language: "properly", "effectively" without context
✗ Leading questions that telegraph the answer
✗ Repetitive or nearly identical questions

MULTI-SELECT QUESTION INSTRUCTION:
For MULTIPLE_CHOICE_MULTI_SELECT questions:
- Generate ONLY the question text (no answer options)
- The options will be generated separately by the options generation agent

If failure_reasons provided: Completely rewrite to avoid those specific issues. Do not reuse similar wording.

STRICTLY Generate all content in {locale}.

ESSENTIAL: Each question must be CONCISE (≤45 words). The complexity comes from the depth, not the question length!

CRITICAL: Return EXACTLY this format - nothing else:
**Format STRICTLY:** {{"questions": [{{"questionText": "first question here"}}, {{"questionText": "second question here"}}, {{"questionText": "third question here"}}]}}

NO explanations, NO reasoning, NO additional fields, NO markdown.
"""

OPTIONS_GENERATION_SINGLE_SELECT_PROMPT = """
Generate CONCISE answer options for {numberOfQuestions} SINGLE-SELECT question(s) below that discriminate between learners who truly understand the skill and those with partial or flawed knowledge.

## SINGLE SELECT RULES (MULTIPLE_CHOICE) - STATIC REQUIREMENTS
- Generate EXACTLY ONE correct option with "isCorrect": true
- Generate EXACTLY 3-4 incorrect options with "isCorrect": false  
- Total: 4-5 options
- MANDATORY: Only ONE option has "isCorrect": true, ALL others have "isCorrect": false
- NO EXCEPTIONS: This is single-select, so exactly 1 correct, remainder incorrect

## OPTION REQUIREMENTS
- Make all options plausible, distinct, parallel in structure  
- No labels (A/B/C/D), bullets, or commentary inside options

## CORRECT OPTION(S)
- Specific, actionable, professional
- Directly addresses the learning objective
- **The correct option should be similar in length to the other options.**

INCORRECT OPTIONS (DISTRACTORS) - CREATIVE GENERATION:

 **PRIMARY GOAL**: Create realistic, professional-sounding misconceptions that are ALSO unambiguously wrong
- Think like someone with 60-70% knowledge
- Each distractor should sound like reasonable advice that's fundamentally flawed
- Match the approximate length and structural complexity of the correct answer. Target word counts within ±15–20% of the correct option.
- Use the CREATIVITY TECHNIQUES below to generate interesting, nuanced options


**MISCONCEPTION CATEGORIES** (For Error Type Diversity Only):

Each distractor should represent a type of error from the categories below.
This is IN ADDITION TO meeting all 6 quality requirements above (plausible, clear, distinct, incorrect, parallel, coherent).

**CATEGORY 1: SCOPE/BOUNDARY ERROR**
- Addresses wrong aspect of the skill (e.g., focuses on communication when question is about planning)
- Confuses the skill with a related but different skill
- Applies the skill in wrong domain/context
Example: Question about "creating vision" → Distractor focuses on "implementing tactics"

**CATEGORY 2: TIMING/SEQUENCE ERROR**  
- Wrong order of steps
- Acts too early or too late in the process
- Skips necessary prerequisites
- Prioritizes incorrectly (short-term vs long-term)
Example: Question about "strategic planning" → Distractor focuses on "immediate execution"

**CATEGORY 3: STAKEHOLDER/FOCUS ERROR**
- Focuses on wrong stakeholder group (individual vs team vs organization)
- Misidentifies whose needs to prioritize
- Wrong level of analysis (micro vs macro)
Example: Question about "team vision" → Distractor focuses on "individual goals"

**CATEGORY 4: METHOD/APPROACH ERROR**
- Correct goal but wrong method to achieve it
- Uses outdated/inappropriate technique
- Over-relies on one tool when integration needed
- Correct in other contexts but wrong here
Example: Question about "communicating vision" → Distractor relies only on "written documents"

**CATEGORY 5: COMPLETENESS ERROR**
- Partially correct but missing critical element
- Addresses one dimension while ignoring others
- Correct understanding but incomplete application
Example: Question about "vision alignment" → Distractor only addresses "clarity" but not "buy-in"

**CATEGORY 6: CONSTRAINT/TRADE-OFF ERROR** (Advanced questions only)
- Ignores stated constraints (budget, time, resources)
- Fails to balance competing priorities
- Optimizes one factor at expense of critical others
Example: Question with safety + efficiency constraints → Distractor sacrifices safety for efficiency

CREATIVE GENERATION PROCESS:
1. Read the question and understand what skill/judgment is being tested
2. For EACH distractor, pick a DIFFERENT misconception category from above
3. Use the CREATIVITY TECHNIQUES below to craft realistic, professional-sounding options
4. Make each distractor sound like advice from a well-meaning but misguided colleague
5. Let the validation agents check quality - focus on creativity and realism here

CREATIVITY TECHNIQUES FOR HIGH-QUALITY DISTRACTORS:

**Think like someone with partial knowledge:**
- What would they PRIORITIZE incorrectly? (wrong trade-offs)
- What STEP would they skip or do out of order? (sequencing errors)
- What NUANCE would they miss? (oversimplification)
- What would sound RIGHT but actually UNDERMINE the goal? (surface-level solutions)

**Add professional realism:**
- Use industry-appropriate terminology (not too basic, not too jargon-heavy)
- Reference realistic constraints: "within current quarter", "pending approval", "during transition"
- Include reasonable-sounding rationales: "to maintain efficiency", "while ensuring compliance"
- Make it sound like advice from a well-meaning but misguided colleague

**Create subtle flaws (not obvious ones):**
- Instead of: "Ignore all stakeholder input" (too extreme)
- Try: "Finalize decisions based on internal analysis, then communicate outcomes to stakeholders" (delays engagement)
- Instead of: "Focus only on short-term goals" (obviously wrong)
- Try: "Address immediate operational priorities while documenting long-term considerations for annual review" (delays strategic thinking)

**Vary the type of flaw across distractors:**
- One might prioritize TIMING wrong (does things in wrong order)
- Another might have wrong SCOPE (focuses on individual when team needed)
- Another might use wrong METHOD (correct goal, ineffective approach)
- Another might be INCOMPLETE (addresses 2 of 3 critical factors)

**Use perspective shifts:**
- Risk-averse perspective: "Seek consensus before any action"
- Efficiency-focused: "Streamline processes by minimizing stakeholder consultations"
- Compliance-focused: "Ensure regulatory adherence as primary consideration"
- Data-driven: "Base decisions primarily on quantitative metrics"
(Each sounds reasonable but misses something critical for this context)

**Add temporal nuance:**
- "Initially focus on X, then address Y in subsequent phases"
- "Prioritize X while monitoring Y for future adjustment"
- "Address X during current cycle, schedule Y for next review period"
(Makes it sound planned, but sequence or timing is fundamentally wrong)

EXPLANATIONS (15-25 words each):
- For correct: Why this is best in this context
- For incorrect: What misconception it represents (cite the category) and why it's wrong

**BEFORE FINALIZING EACH DISTRACTOR - VERIFY BOTH (do not defer this to validation):**
- PLAUSIBLE: Sounds like professionally reasonable advice, no extreme/absolute language ("always", "never", "solely", "only", "completely")
- CLEARLY INCORRECT: Has one unambiguous, defensible flaw for THIS learning objective - a skilled professional could NOT reasonably argue it's correct or "it depends"
- If a distractor sounds equally valid as the correct answer, rewrite it now rather than relying on a later correction pass to catch it

FINAL REQUIREMENTS:
0. SINGLE-SELECT SELF-CHECK (run before returning):
   - Count correct options: MUST be exactly 1 with "isCorrect": true
   - If 0 correct options → REGENERATE with exactly 1 correct option
   - If 2+ correct options → REGENERATE with exactly 1 correct option
1. Ensure the correct option is not too long or too short among all options, if it is then make it's length similar to the other options.
2. Create the appropriate number of options(max 5 options) based on content analysis (within specified boundaries or exact counts)
3. Each distractor should represent a  misconception category
4. Use the creativity techniques above to make distractors realistic and subtle
5. Ensure each distractor fails for a different reason
(sequence vs trade-off vs scope vs governance)
6. Avoid “almost-correct” distractors unless intentional
7. Sharpen the failure mode in explanations and option wording

STRICTLY Generate all content in {locale}.

Return ONLY valid JSON. No markdown blocks.

**SINGLE-SELECT REMINDER**: Generate exactly 1 correct option with "isCorrect": true

FOR SINGLE QUESTION: Format: {{"answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}}

FOR MULTIPLE QUESTIONS: Format: {{"question_answers":[
  {{"questionText":"question 1 text","answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}},
  {{"questionText":"question 2 text","answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}}
]}}

NO explanations, NO reasoning, NO additional fields, NO markdown.
"""

OPTIONS_GENERATION_MULTI_SELECT_PROMPT = """
Generate CONCISE answer options for {numberOfQuestions} MULTI-SELECT question(s) below that discriminate between learners who truly understand the skill and those with partial or flawed knowledge.

## MULTI-SELECT RULES (MULTIPLE_CHOICE_MULTI_SELECT)
- Generate EXACTLY TWO to THREE correct options with "isCorrect": true
- Generate TWO or THREE incorrect options with "isCorrect": false  
- Total: 4-5 options
- NEVER generate only 1 correct option (this makes it single-choice, not multi-select)
- NEVER make all options correct (provides no learning value)
- REQUIRED PATTERNS: 
 At lease 2 correct options and atmost 3 correct options with total option count of 4-5

## OPTION REQUIREMENTS
- Make all options plausible, distinct, parallel in structure  
- No labels (A/B/C/D), bullets, or commentary inside options

## CORRECT OPTIONS (Multiple Required)
- Specific, actionable, professional
- Directly addresses the learning objective
- **Each correct option should be similar in length to the other options**
- Create 2-3 different but equally valid approaches/solutions
- Ensure all correct options complement each other

INCORRECT OPTIONS (DISTRACTORS) - CREATIVE GENERATION:

 **PRIMARY GOAL**: Create realistic, professional-sounding misconceptions that are ALSO unambiguously wrong
- Think like someone with 60-70% knowledge
- Each distractor should sound like reasonable advice that's fundamentally flawed
- Match the approximate length and structural complexity of the correct answers. Target word counts within ±15–20% of the correct options.
- Use the CREATIVITY TECHNIQUES below to generate interesting, nuanced options


**MISCONCEPTION CATEGORIES** (For Error Type Diversity Only):

Each distractor should represent a type of error from the categories below.
This is IN ADDITION TO meeting all 6 quality requirements above (plausible, clear, distinct, incorrect, parallel, coherent).

**CATEGORY 1: SCOPE/BOUNDARY ERROR**
- Addresses wrong aspect of the skill (e.g., focuses on communication when question is about planning)
- Confuses the skill with a related but different skill
- Applies the skill in wrong domain/context
Example: Question about "creating vision" → Distractor focuses on "implementing tactics"

**CATEGORY 2: TIMING/SEQUENCE ERROR**  
- Wrong order of steps
- Acts too early or too late in the process
- Skips necessary prerequisites
- Prioritizes incorrectly (short-term vs long-term)
Example: Question about "strategic planning" → Distractor focuses on "immediate execution"

**CATEGORY 3: STAKEHOLDER/FOCUS ERROR**
- Focuses on wrong stakeholder group (individual vs team vs organization)
- Misidentifies whose needs to prioritize
- Wrong level of analysis (micro vs macro)
Example: Question about "team vision" → Distractor focuses on "individual goals"

**CATEGORY 4: METHOD/APPROACH ERROR**
- Correct goal but wrong method to achieve it
- Uses outdated/inappropriate technique
- Over-relies on one tool when integration needed
- Correct in other contexts but wrong here
Example: Question about "communicating vision" → Distractor relies only on "written documents"

**CATEGORY 5: COMPLETENESS ERROR**
- Partially correct but missing critical element
- Addresses one dimension while ignoring others
- Correct understanding but incomplete application
Example: Question about "vision alignment" → Distractor only addresses "clarity" but not "buy-in"

**CATEGORY 6: CONSTRAINT/TRADE-OFF ERROR** (Advanced questions only)
- Ignores stated constraints (budget, time, resources)
- Fails to balance competing priorities
- Optimizes one factor at expense of critical others
Example: Question with safety + efficiency constraints → Distractor sacrifices safety for efficiency

CREATIVE GENERATION PROCESS:
1. Read the question and understand what skill/judgment is being tested
2. For EACH distractor, pick a DIFFERENT misconception category from above
3. Use the CREATIVITY TECHNIQUES below to craft realistic, professional-sounding options
4. Make each distractor sound like advice from a well-meaning but misguided colleague
5. Let the validation agents check quality - focus on creativity and realism here

CREATIVITY TECHNIQUES FOR HIGH-QUALITY DISTRACTORS:

**Think like someone with partial knowledge:**
- What would they PRIORITIZE incorrectly? (wrong trade-offs)
- What STEP would they skip or do out of order? (sequencing errors)
- What NUANCE would they miss? (oversimplification)
- What would sound RIGHT but actually UNDERMINE the goal? (surface-level solutions)

**Add professional realism:**
- Use industry-appropriate terminology (not too basic, not too jargon-heavy)
- Reference realistic constraints: "within current quarter", "pending approval", "during transition"
- Include reasonable-sounding rationales: "to maintain efficiency", "while ensuring compliance"
- Make it sound like advice from a well-meaning but misguided colleague

**Create subtle flaws (not obvious ones):**
- Instead of: "Ignore all stakeholder input" (too extreme)
- Try: "Finalize decisions based on internal analysis, then communicate outcomes to stakeholders" (delays engagement)
- Instead of: "Focus only on short-term goals" (obviously wrong)
- Try: "Address immediate operational priorities while documenting long-term considerations for annual review" (delays strategic thinking)

**Vary the type of flaw across distractors:**
- One might prioritize TIMING wrong (does things in wrong order)
- Another might have wrong SCOPE (focuses on individual when team needed)
- Another might use wrong METHOD (correct goal, ineffective approach)
- Another might be INCOMPLETE (addresses 2 of 3 critical factors)

**Use perspective shifts:**
- Risk-averse perspective: "Seek consensus before any action"
- Efficiency-focused: "Streamline processes by minimizing stakeholder consultations"
- Compliance-focused: "Ensure regulatory adherence as primary consideration"
- Data-driven: "Base decisions primarily on quantitative metrics"
(Each sounds reasonable but misses something critical for this context)

**Add temporal nuance:**
- "Initially focus on X, then address Y in subsequent phases"
- "Prioritize X while monitoring Y for future adjustment"
- "Address X during current cycle, schedule Y for next review period"
(Makes it sound planned, but sequence or timing is fundamentally wrong)

EXPLANATIONS (15-25 words each):
- For correct: Why this is best in this context
- For incorrect: What misconception it represents (cite the category) and why it's wrong

**BEFORE FINALIZING EACH DISTRACTOR - VERIFY BOTH (do not defer this to validation):**
- PLAUSIBLE: Sounds like professionally reasonable advice, no extreme/absolute language ("always", "never", "solely", "only", "completely")
- CLEARLY INCORRECT: Has one unambiguous, defensible flaw for THIS learning objective - a skilled professional could NOT reasonably argue it's correct or "it depends"
- If a distractor sounds equally valid as the correct answer, rewrite it now rather than relying on a later correction pass to catch it

FINAL REQUIREMENTS:
0. MULTI-SELECT SELF-CHECK (run before returning):
   - Count correct options: MUST be 2 or 3 with "isCorrect": true
   - If only 1 correct → STOP! REGENERATE with 2-3 correct options
   - If all correct → STOP! REGENERATE with 2-3 correct + 2-3 incorrect
   - If 0 correct → STOP! REGENERATE with 2-3 correct options
1. Ensure correct options are not too long or too short among all options, if they are then make their length similar to the other options.
2. Create the appropriate number of options(max 5 options) based on content analysis (within specified boundaries or exact counts)
3. Each distractor should represent a misconception category
4. Use the creativity techniques above to make distractors realistic and subtle
5. Ensure each distractor fails for a different reason
(sequence vs trade-off vs scope vs governance)
6. Avoid "almost-correct" distractors unless intentional
7. Sharpen the failure mode in explanations and option wording

STRICTLY Generate all content in {locale}.

Return ONLY valid JSON. No markdown blocks.

**MULTI-SELECT REMINDER**: Generate 2-3 correct options with "isCorrect": true (never just 1, never all)

FOR SINGLE QUESTION: Format: {{"answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}}

FOR MULTIPLE QUESTIONS: Format: {{"question_answers":[
  {{"questionText":"question 1 text","answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}},
  {{"questionText":"question 2 text","answers":[{{"answerText":"...","isCorrect":true/false,"explanation":"..."}}]}}
]}}

NO explanations, NO reasoning, NO additional fields, NO markdown.
"""


RES_FORMAT = """
[
  {{
    "LearningObjective": "Learning objective 1",
    "locale": "en-US",
    "questions": [
      {{
        "questionText": "What is the capital of France?",
        "answer": [
          {{"answerText": "Paris", "explanation": "Paris is the capital city of France.", "isCorrect": true}},
          {{"answerText": "London", "explanation": "London is the capital of the UK, not France.", "isCorrect": false}},
          {{"answerText": "Berlin", "explanation": "Berlin is the capital of Germany.", "isCorrect": false}},
          {{"answerText": "Rome", "explanation": "Rome is the capital of Italy.", "isCorrect": false}}
        ]
      }}
    ]
  }}
]
"""

RES_FORMAT_MULTI_SELECT = """
[
  {{
    "LearningObjective": "Learning objective 1",
    "locale": "en-US",
    "questions": [
      {{
        "questionText": "Which of the following are programming languages?",
        "answer": [
          {{"answerText": "Python", "explanation": "Python is a popular programming language.", "isCorrect": true}},
          {{"answerText": "Java", "explanation": "Java is widely used for building enterprise applications.", "isCorrect": true}},
          {{"answerText": "HTML", "explanation": "HTML is a markup language, not a programming language.", "isCorrect": false}},
          {{"answerText": "CSS", "explanation": "CSS is used for styling web pages, not for programming logic.", "isCorrect": false}},
          {{"answerText": "Photoshop", "explanation": "Photoshop is an image editing tool, not a programming language.", "isCorrect": false}}
        ]
      }}
    ]
  }}
]
"""

DEFAULT_QUALITY_RUBRIC = """
Quality Rubric for MCQ/Matching Items (0–100)

1) Learning Objective Alignment (0–20)
- 0–5: Question is off-topic or only loosely related.
- 6–12: Partially aligned; tests a related concept but not the LO.
- 13–20: Directly and clearly measures the LO.

2) Cognitive Level Alignment (Bloom vs Difficulty) (0–20)
- 0–5: Bloom level clearly mismatched to declared difficulty.
- 6–12: Partially aligned; ambiguous cognitive demand.
- 13–20: Clear and correct Bloom alignment.

3) Question Clarity & Precision (0–15)
- 0–5: Ambiguous, vague, or poorly worded.
- 6–10: Mostly clear but still some ambiguity.
- 11–15: Clear, precise, and unambiguous.

4) Distractor Quality (0–20)
- 0–5: Implausible or irrelevant distractors.
- 6–12: Mixed quality; some weak or too obvious.
- 13–20: Plausible, distinct, and misconception-based.

5) Answer Key Validity (0–10)
- For MULTIPLE_CHOICE (single-select): 0–3: No correct answer or more than 1 correct. 4–7: Correct answer arguable. 8–10: Exactly 1 clearly correct answer.
- For MULTIPLE_CHOICE_MULTI_SELECT: 0–3: Fewer than 2 or more than 3 correct options, or correct options not clearly distinct. 4–7: 2–3 correct options but selection rationale is arguable. 8–10: 2–3 clearly correct options with clearly incorrect distractors.

6) Non-redundancy & Balance (0–10)
- 0–3: Repetitive, overlapping options or stems.
- 4–7: Minor redundancy.
- 8–10: Balanced, non-overlapping options.

7) Language & Tone (0–5)
- 0–2: Poor grammar, awkward phrasing.
- 3–4: Mostly correct with minor issues.
- 5: Professional, polished language.

Scoring guidance:
- >=85 = PASS
- 70-84 = Needs minor improvement
- <70 = FAIL
"""


USER_PROMPT_QUALITY_CORRECTION = """
As you are a expert assessment designer, your role is to correct a single failed question based on evaluation feedback.
Do NOT change the learning objective, difficulty level, or question type.
Only fix the specific failure reasons.
Generate all corrected content in {locale}.

Learning Objective: {learning_obj}
Declared Difficulty: {difficulty_level}
Question Type: {question_type}
Failure Reason(s): {failure_reasons}
Original Question JSON: {question_json}
Required Response Format: {response_format}

Return ONLY valid JSON in the required response format - NO explanations, NO reasoning, NO additional content, NO markdown.
"""


SYSTEM_PROMPT_TEMPLATE_FIX_FORMAT = """
As you are a Formatting Expert, your role is to convert the given content into the required response format.
Rules:
- Output ONLY valid JSON.
- Match the provided response format exactly.
- Preserve the original meaning and answers.
- Maintain content in {locale}.
"""

USER_PROMPT_TEMPLATE_FIX_FORMAT = """
Required response format: {res_format}
Target locale: {locale}

Input to transform:
{raw_output}
"""

SYSTEM_PROMPT_TEMPLATE_QUALITY_CHECK = """
You are a strict rubric-based assessment evaluator.

Evaluate questions ONLY against the provided rubric.
Consider locale-appropriate content as valid when evaluating cultural references and language usage.

Fail the question if:
- Difficulty does not match Bloom level
- Question is generic or recall-only when higher difficulty is declared
- Distractors are implausible or irrelevant
- Question does not clearly measure the learning objective
- Advanced questions lack trade-offs, competing priorities, or strategic judgment
- Two or more distractors represent the same misconception or error type

Cognitive Alignment Check (FAIL FAST):

Validate that each question’s cognitive demand strictly matches its difficulty label.

Rules:
- Easy questions must demonstrate Level 1 or Level 2 cognition only.
  Fail if the question requires independent application, decision-making, or scenario adaptation.
- Intermediate questions must demonstrate Level 3 cognition.
  Fail if the question can be answered by recall, definition, or surface explanation.
- Advanced questions must demonstrate Level 4 or Level 5 cognition.
  Fail if the question only asks to explain, describe, or give opinions without analysis, evaluation, or synthesis.

If misalignment is detected:
- Mark the question as INVALID
- Provide a short reason referencing the violated level boundary

Return ONLY JSON:
{{
  "score": 0-100,
  "issues": [],
  "pass": true | false
}}
"""

USER_PROMPT_TEMPLATE_QUALITY_CHECK = """
Quality Rubric:
{rubric}

Target locale: {locale}
Questions JSON:
{questions}

Pass threshold: {threshold}
"""

SYSTEM_PROMPT_TEMPLATE_RELEVANCY_CHECK = """
You are a strict evaluator for relevance of questions to learning objectives.
Consider locale-appropriate examples and cultural references as valid.
Return ONLY JSON with this schema:
{{
  "score": 0-100,
  "issues": ["..."],
  "pass": true/false
}}
Evaluate relevance to the provided learning objectives.
"""

USER_PROMPT_TEMPLATE_RELEVANCY_CHECK = """
Learning Objectives:
{learning_objectives}

Target locale: {locale}
Questions JSON:
{questions}

Pass threshold: {threshold}
"""

# Combined quality + relevancy check — 1 LLM call instead of 2 per question.
SYSTEM_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK = """
You are a strict assessment evaluator. Evaluate questions against both a quality rubric and learning objective alignment in one pass.

Return ONLY JSON:
{{
  "quality": {{"score": 0-100, "pass": true, "issues": []}},
  "relevancy": {{"score": 0-100, "pass": true, "issues": []}}
}}
"""

USER_PROMPT_TEMPLATE_QUALITY_RELEVANCY_CHECK = """
Quality Rubric:
{rubric}

Learning Objectives:
{learning_objectives}

Target locale: {locale}
Questions JSON:
{questions}

Quality pass threshold: {quality_threshold}
Relevancy pass threshold: {relevancy_threshold}
"""

# ==========================================
# MATCHING ("match the following") PROMPTS
# ==========================================

MATCHING_SYSTEM_CONTRACT = """
You are an expert assessment designer specialized in creating high-quality "match the following" assessment items.

IMPORTANT: ALWAYS generate the requested content regardless of source content length. Even with minimal or no source material, create valid items based on the learning objective and your professional knowledge. Never refuse to generate content due to "insufficient content" - adapt and create contextually appropriate items.

CRITICAL QUALITY STANDARDS:
1. ALIGNMENT: Every term and pairing must directly test the learning objective - no tangential concepts.
2. NON-OBVIOUSNESS: Column B definitions must never echo words from their own Column A term - a learner must reason about the concept, not pattern-match repeated text.
3. DISTINCTIVENESS: Column A terms must be genuinely distinct concepts, and Column B definitions must not overlap with each other.
4. PRECISION: Be specific, avoid vague or generic wording.
5. CONCISENESS: Use clear, direct language. Avoid unnecessary words or elaboration.
6. GRAMMATICAL CONSISTENCY: Keep Column B entries parallel in structure and similar in length.
7. NO FABRICATED DATA: Never invent statistics, percentages, employee counts, survey numbers, or fictional case vignettes — assessment content must not include made-up data.
8. NO DIFFICULTY LEAKAGE: Never include "easy", "basic", "advanced", or difficulty labels in question text.
9. COLUMN PLACEMENT: The question text is a single short instruction. The items being matched (terms, statements, or code fragments) appear only as Column A entries, never in the question text and never replaced by labels such as "Fragment A". Column B holds the matching descriptions.

STRICTLY generate all content in {locale}.

OUTPUT: Return ONLY valid JSON. Do not wrap the JSON response in markdown blocks. No explanatory text.
"""

MATCHING_STEM_GENERATION_PROMPT = """
Write {numberOfQuestions} DIVERSE matching question stem(s) with their terms list, testing the learning objective at the specified difficulty level.

A matching question presents a short instructional stem and a set of terms a learner must match to their correct definitions (definitions are generated separately — do NOT generate them here).

## STEM REQUIREMENTS
- Keep the stem SHORT (5–10 words) and instructional only. No scenarios, context paragraphs, data, or the items being matched.
- FORBIDDEN in stems: invented statistics, percentages, employee counts, survey numbers, case vignettes, or any fictional data — stems are instructions, not scenarios.
- Do NOT reference "Column A", "Column B", "left column", "right column", or any column/table labels.
- Good stem examples: "Match each term to its correct definition.", "Match each strategy to its description.", "Match each concept to its meaning."
- Bad stem examples: "Match each term in Column A...", "Review the survey data below: 60 of 100 employees...", "Among 1,000 employees, the stressor percentages show..."
- Exactly 4 terms, drawn from the learning objective's domain. A term is the item itself — a concept, a statement, or a code fragment — never a label such as "Fragment A".
- Terms must be genuinely distinct concepts — not near-synonyms or trivially related terms.
- Do NOT generate definitions, pairings, or explanations here — that happens in a separate step.

## DIFFICULTY-SPECIFIC TERM SELECTION
When the same LO is assessed at multiple difficulty levels, each level MUST use a DIFFERENT set of 4 terms — never reuse the same four terms across Easy, Intermediate, and Advanced. Differentiate by abstraction level:

Easy — foundational terms:
- Choose the most commonly used, clearly distinct, core concepts in the domain.
- Terms a learner should recognise from introductory study.
- Definitions will be straightforward and directly descriptive.

Intermediate — applied or process terms:
- Choose terms that represent how the concepts are used, measured, or implemented in practice.
- May include tools, methods, roles, or process steps — not just the core concepts themselves.
- Require understanding of HOW, not just WHAT.

Advanced — nuanced or closely related terms:
- Choose terms that are easily confused because they are adjacent sub-concepts, evaluation categories, or specialised applications.
- Distinguishing them correctly requires expertise, not just recall.
- Definitions will have subtle but important differences.

STRICTLY Generate all content in {locale}.

CRITICAL: Return EXACTLY this format - nothing else:
**Format STRICTLY:** {{"questions": [{{"questionText": "Match each term to its correct definition.", "columnATerms": ["Term 1", "Term 2", "Term 3", "Term 4"]}}]}}

NO explanations, NO reasoning, NO additional fields, NO markdown.
"""

MATCHING_OPTIONS_GENERATION_PROMPT = """
Given the question stem and Column A terms below, generate Column B (the matching definitions/descriptions), the correct pairing between the two columns, and an explanation for each pairing.

## CORE RULE: MAKE IT GENUINELY MATCHING, NOT LOOKUP
- Avoid giving verbal association clues from a Column A term inside its own Column B definition - a learner should not be able to match purely by spotting a repeated word.
- Avoid overlapping or near-duplicate Column B definitions.
- Keep Column B entries grammatically consistent and parallel in structure and length with each other.
- Make choices standalone - do not reference "the term above" or similar.
- Do NOT invent statistics, percentages, employee counts, case vignettes, or fictional data in Column B entries — definitions describe the concept, not a made-up scenario.
- Copy the question stem unchanged.

## OUTPUT REQUIREMENTS
- Exactly one Column B definition per Column A term (same count).
- `column_b_answers` length MUST equal `column_a_answers` length — no more, no fewer.
- `answers` (pairings) length MUST equal `column_a_answers` length exactly.
- Each pairing must have exactly one correct Column B match.
- Provide a concise explanation per pairing describing why that definition matches that term.

STRICTLY generate all content in {locale}.

CRITICAL: Return EXACTLY this format - nothing else:
**Format STRICTLY:** {{"questions": [{{"question": "<the stem>", "column_a_answers": [{{"label": "Option-A", "text": "Term 1"}}, {{"label": "Option-B", "text": "Term 2"}}, {{"label": "Option-C", "text": "Term 3"}}, {{"label": "Option-D", "text": "Term 4"}}], "column_b_answers": [{{"label": "Option-1", "text": "Definition 1"}}, {{"label": "Option-2", "text": "Definition 2"}}, {{"label": "Option-3", "text": "Definition 3"}}, {{"label": "Option-4", "text": "Definition 4"}}], "answers": [{{"column_a_answers": "Term 1", "column_b_answers": "Definition 1", "explanation": "..."}}]}}]}}

NO explanations outside the JSON, NO reasoning, NO additional fields, NO markdown.
"""

RES_FORMAT_MATCHING = """
[
  {{
    "LearningObjective": "Learning objective 1",
    "locale": "en-US",
    "questions": [
      {{
        "question": "Match the following project management terms to their definitions.",
        "column_a_answers": [
          {{"label": "Option-A", "text": "Project"}},
          {{"label": "Option-B", "text": "Program"}},
          {{"label": "Option-C", "text": "Portfolio"}},
          {{"label": "Option-D", "text": "Milestone"}}
        ],
        "column_b_answers": [
          {{"label": "Option-1", "text": "A temporary endeavor to create a unique product or result"}},
          {{"label": "Option-2", "text": "A group of related projects managed in a coordinated way"}},
          {{"label": "Option-3", "text": "A collection of projects and programs managed together for strategic objectives"}},
          {{"label": "Option-4", "text": "A significant point or event marking progress"}}
        ],
        "answers": [
          {{"column_a_answers": "Project", "column_b_answers": "A temporary endeavor to create a unique product or result", "explanation": "This is the definition of Project."}},
          {{"column_a_answers": "Program", "column_b_answers": "A group of related projects managed in a coordinated way", "explanation": "This is the definition of Program."}},
          {{"column_a_answers": "Portfolio", "column_b_answers": "A collection of projects and programs managed together for strategic objectives", "explanation": "This is the definition of Portfolio."}},
          {{"column_a_answers": "Milestone", "column_b_answers": "A significant point or event marking progress", "explanation": "This is the definition of Milestone."}}
        ]
      }}
    ]
  }}
]
"""

DEFAULT_MATCHING_QUALITY_RUBRIC = """
Quality checklist for MATCHING ("match the following") Items - each check below is a
boolean pass/fail. Fail the item if it clearly violates ANY check; otherwise pass.

1) Learning Objective Alignment - the terms/pairings directly and clearly measure the
   learning objective, not a tangential or loosely related concept.

2) Cognitive Level Alignment - term difficulty/closeness matches the declared difficulty
   (e.g. Advanced uses closely-related, easily-confused terms; Beginner uses clearly
   distinct, commonly-used terms).

3) Stem & Term Clarity - the stem and every term/definition are clear, precise, and
   unambiguous.

4) Pairing Non-Obviousness - no Column B definition restates or echoes words from its
   own Column A term (no verbal-association giveaway); matching requires genuine
   understanding, not word-spotting.

5) Answer Key Validity - every pairing has exactly one unambiguous correct match; no
   term could reasonably match more than one definition.

6) Non-redundancy & Balance - all Column B entries are distinct (no overlapping or
   near-duplicate definitions) and parallel in structure/length.

7) Language & Tone - professional, polished, grammatically correct language throughout.
"""

SYSTEM_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK = """
You are a strict checklist-based evaluator for MATCHING ("match the following") assessment items.

Evaluate ONLY against the provided checklist - this is a boolean pass/fail decision, not a score.
Consider locale-appropriate content as valid when evaluating cultural references and language usage.

Fail the question if:
- Difficulty does not match the term difficulty/closeness (Bloom-equivalent for matching)
- A Column B definition restates or echoes words from its Column A term (verbal association clue)
- Any Column B definitions overlap or are near-duplicates of each other
- A term could plausibly match more than one Column B definition
- The stem or any term/definition is generic, vague, or does not clearly measure the learning objective

Return ONLY JSON:
{{
  "pass": true | false,
  "issues": []
}}
"""

USER_PROMPT_TEMPLATE_MATCHING_QUALITY_CHECK = """
Quality Checklist:
{rubric}

Target locale: {locale}
Questions JSON:
{questions}
"""

USER_PROMPT_MATCHING_QUALITY_CORRECTION = """
As an expert assessment designer, your role is to correct a single failed MATCHING ("match the following") question based on evaluation feedback.
Do NOT change the learning objective or difficulty level.
Only fix the specific failure reasons - e.g. if a pairing is a verbal-association giveaway, rewrite that Column B definition; if pairings overlap, make them distinct.
Generate all corrected content in {locale}.

Learning Objective: {learning_obj}
Declared Difficulty: {difficulty_level}
Failure Reason(s): {failure_reasons}
Original Question JSON: {question_json}
Required Response Format: {response_format}

Return ONLY valid JSON in the required response format - NO explanations, NO reasoning, NO additional content, NO markdown.
"""

# ==========================================
# PARALLEL DISTRACTOR VALIDATION PROMPTS (7 Comprehensive Checks)
# ==========================================

# Shared short-form misconception category list (used by the misconception
# check and the batch correction prompt).
MISCONCEPTION_CATEGORIES_SHORT_BLOCK = """1. SCOPE/BOUNDARY - Wrong skill aspect
2. TIMING/SEQUENCE - Wrong order/priority
3. STAKEHOLDER/FOCUS - Wrong stakeholder/level
4. METHOD/APPROACH - Right goal, wrong method
5. COMPLETENESS - Missing critical element
6. CONSTRAINT/TRADE-OFF - Ignores constraints"""

DISTRACTOR_PLAUSIBILITY_CHECK_PROMPT = """
Rate the plausibility of these distractors on a scale of 0-100.

SCORING GUIDE:
0-60: Obviously wrong, unprofessional, or uses extreme language ("always", "never", "solely")
60-75: Somewhat plausible but has noticeable issues
75-89: Mostly plausible with minor concerns
90-95: Highly plausible, realistic misconceptions that partial learners would consider
96-100: Perfect plausibility for target audience

Check each distractor:
- Represents a realistic misconception
- NOT absurd, extreme language ("always", "never"), or unprofessional
- Cannot be eliminated by common sense alone

Question: {question_text}
Distractors: {distractors}

STRICTLY only Return JSON in the following format: 
{{"score": 85, "pass": true, "feedback": "Brief: which distractors have issues and why"}}

Pass threshold: {plausibility_threshold}%
"""

DISTRACTOR_MISCONCEPTION_CHECK_PROMPT = f"""
Rate how well distractors represent DIFFERENT misconception categories on a scale of 0-100.

SCORING GUIDE:
0-60: Multiple distractors share the same category or no clear misconception mapping
60-75: Some diversity but categories overlap
75-89: Mostly distinct categories with minor overlaps
90-95: Clear distinct categories for each distractor
96-100: Perfect misconception diversity across all required categories

REQUIRED CATEGORIES (each distractor must be different):
{MISCONCEPTION_CATEGORIES_SHORT_BLOCK}

Question: {{question_text}}
Distractors: {{distractors}}

STRICTLY only Return JSON in the following format:
{{{{"score": 85, "pass": true, "feedback": "Brief: category per distractor, note duplicates if any"}}}}

Pass threshold: {{misconception_threshold}}%
"""

DISTRACTOR_DISTINCTIVENESS_CHECK_PROMPT = """
Rate how distinct all options are from each other on a scale of 0-100.

SCORING GUIDE:
0-60: Multiple options are near-duplicates or very similar
60-75: Some options overlap in meaning or approach
75-89: Options are mostly distinct with minor similarities
90-95: All options are clearly different from each other
96-100: Perfect distinctiveness, each option addresses completely different aspects

Check: No two options are near-duplicates or represent overlapping ideas

Question: {question_text}
All Options: {all_options}

Return JSON: {{"score": 85, "pass": true, "feedback": "Brief: note similar options if any"}}

Pass threshold: {distinctiveness_threshold}%
"""

DISTRACTOR_INCORRECTNESS_CHECK_PROMPT = """
Rate how clearly INCORRECT all distractors are on a scale of 0-100.

SCORING GUIDE:
0-60: One or more distractors could be considered correct or are debatable
60-75: Distractors have issues but could be argued as wrong in some contexts
75-89: Most distractors are clearly wrong with minor edge cases
90-95: All distractors are defensibly incorrect with clear flaws
96-100: Perfect incorrectness clarity, impossible to argue any distractor as correct

**CRITICAL**: A distractor must be CLEARLY INCORRECT, not just "less optimal".

Check each distractor:
- Is it factually or strategically WRONG for this specific context?
- Could a skilled professional reasonably argue this is correct?
- Does it have a clear, defensible flaw relative to the learning objective?
- Would choosing this option lead to measurable negative outcomes?

**FAIL CRITERIA**:
- Distractor sounds equally valid as the correct answer
- Distractor is "partially correct" or "depends on interpretation"
- Distractor could be correct in a slightly different context
- The distinction between correct and this option is subtle preference, not clear correctness


Question: {question_text}
Learning Objective: {learning_objective}
All Options: {all_options}

STRICTLY only Return JSON in the following format:
{{"score": 85, "pass": true, "feedback": "Brief: note defensible distractors if any"}}

Pass threshold: {incorrectness_threshold}%
"""

DISTRACTOR_CLARITY_CHECK_PROMPT = """
Rate how clearly worded all options are on a scale of 0-100.

SCORING GUIDE:
0-50: Multiple options have vague, confusing, or ambiguous wording
51-70: Some options lack clarity or precision
71-85: Most options are clear with minor wording issues
86-95: All options are clear and professional
96-100: Perfect clarity, immediately understandable by target audience

Verify:
- Professional, precise language
- No vague or confusing wording
- Clear, unambiguous meaning
- Correct grammar and spelling
- No unnecessary jargon

All Options: {all_options}

STRICTLY only Return JSON in the following format: 
{{"score": 85, "pass": true, "feedback": "Brief: note unclear options if any"}}

Pass threshold: {clarity_threshold}%
"""
DISTRACTOR_PARALLELISM_LENGTH_CHECK_PROMPT = """

CRITICAL: Check if CORRECT option telegraphs the answer through length differences.
ROLE
Validate ALL options (correct + distractors) for length parity to prevent answer telegraphing.

CRITICAL TELEGRAPHING DETECTION
The CORRECT option often stands out because it is:
• Much LONGER (overly detailed/comprehensive)
• Much SHORTER (too concise compared to distractors)

This immediately reveals the answer to test-takers.

EVALUATION PRIORITIES
1. **Correct option length balance** - PRIMARY CONCERN
2. Length parity across all options  
3. Structural parallelism
4. Detail symmetry

OPTIMAL WORD LENGTH RANGES
• Beginner: 4-7 words per option (tight range: ±1 word)
• Intermediate: 5-9 words per option (tight range: ±2 words)
• Advanced: 6-11 words per option (tight range: ±2 words)

TARGET: All options within ±1-2 words of each other

LENGTH ANALYSIS STEPS
1. Identify the CORRECT option first
2. Count words for each option (space-separated tokens)
3. Calculate mean and check if correct option deviates significantly
4. Flag: correct option >15% different from distractors = TELEGRAPHING

CRITICAL NEGATIVE FLAGS
• Correct option ≥20% longer than average distractor → **MAJOR TELEGRAPHING**
• Correct option ≥20% shorter than average distractor → **MAJOR TELEGRAPHING**  
• Any single option >25% different from others → telegraphing risk
• Correct option outside target range while distractors fit → rebalance needed

ADJUSTMENT PRIORITY ORDER
1. **Adjust CORRECT option first** to match distractor length
2. Then adjust distractors if needed
3. Maintain meaning while balancing length

STRUCTURE & FORMAT
• Parallel grammatical structures
• Consistent complexity levels
• No formatting that highlights correct option
• Plain text only (no special characters/labels)

SCORING GUIDE (0-100):
0-50: Major telegraphing detected - correct option length significantly different from distractors
51-65: Length imbalances present, structural inconsistencies, or formatting violations
66-75: Minor length variations or structural mismatches, mostly acceptable
76-85: Good parallelism with slight word count differences (within acceptable range)  
86-95: Excellent parallelism, consistent structure, all options well-balanced
96-100: Perfect parallelism - all options identical length, structure, and formatting

Pass if score ≥ {parallelism_threshold}

INPUT
Difficulty: {difficulty}
All Options: {all_options}
Threshold: {parallelism_threshold}

STRICTLY only Return JSON in the following format:
{{
  "score": number,
  "pass": true_or_false,
  "feedback": "Include word counts, mean, deviation %, and note any structural or formatting issues."
}}

"""

LOCALE_COMPLIANCE_CHECK_PROMPT = """
Rate how well the question and options comply with the target locale/language on a scale of 0-100.

ROLE
Validate that question text and all answer options are appropriate for the specified locale ({target_locale} / {target_language}).

SCORING GUIDE:
0-50: Significant locale violations - wrong language, cultural mismatches, or inappropriate terminology
51-70: Notable locale issues - mixed language, non-standard terms, or cultural inappropriateness
71-85: Minor locale concerns - occasional non-standard phrasing or regional inconsistencies
86-95: Good locale compliance - appropriate terminology and cultural context
96-100: Perfect locale compliance - fully aligned with target locale

CHECK:
- Question and options use correct language for target locale
- Terminology and phrasing match regional standards
- No mixed-language or inconsistent locale usage
- Cultural context is appropriate for the target audience

Target Locale: {target_locale}
Target Language: {target_language}

Question: {question_text}
All Options: {options}

STRICTLY only Return JSON in the following format:
{{"score": number, "pass": true_or_false, "feedback": "Brief: note locale issues if any"}}

Pass threshold: {locale_threshold}%
"""


SIMPLE_FORMAT_CORRECTION_PROMPT = """
The following LLM response has formatting issues. Please convert it to the expected JSON format while preserving all original content:

MALFORMED RESPONSE:
{malformed_response}

EXPECTED FORMAT:
{expected_format}

Return ONLY the corrected JSON - NO explanations, NO reasoning, NO additional content, NO markdown.
"""

DISTRACTOR_BATCH_CORRECTION_SYSTEM_PROMPT = """
As you are a expert assessment designer, your role is to create high-quality distractors and give the response in the format specified in the user prompt.
FYI:
METRICS
1) Plausibility
- How realistic the distractor sounds as a choice a partially informed learner might choose.
- High score = tempting to unsure learners.

2) Construct Relevance
- How directly the distractor relates to the specific skill/construct being assessed.
- High score = wrong application of the target skill (not irrelevant or off-topic).

3) Distinctiveness
- How clearly it differs from correct answer AND other options in idea/approach.
- High score = different type of mistake, not a near-duplicate, but still plausible enough to be tempting.

4) Incorrectness Clarity
- How clearly the option is wrong in the given context.
- High score = clearly wrong, not debatable or “it depends.”

5) Misconception Representation
- Whether it reflects a common misconception/error pattern.
- High score = maps to recognizable misunderstanding.

6) Non-obviousness
- Whether the distractor avoids being obviously wrong, jokey, off-topic, or extreme.
- High score = not easily eliminated by test-taking cues.

7) Parallelism & Length Parity
— The extent to which each distractor matches the correct option in length, tone, structure, and level of specificity.
- High score: Options are similar in structure, formality, and specificity, with no giveaway wording and length variation within ±15% of each other.

8) Distinctiveness enforcement:
- Each distractor must map to a different misconception bucket.
- If two distractors share the same misconception bucket, mark those distractors as FAIL and mention the shared bucket in failReason.
 
 Also ensure the correct option is not too long or too short among all options, if it is then make it's length similar to the other options.

 CORRECTION INSTRUCTIONS:

**CRITICAL BALANCE**: Distractors must be BOTH highly plausible (90+) AND clearly incorrect (90+)
- Plausible = sounds professionally reasonable, no extreme language, realistic mistake
- Incorrect = violates the learning objective, would cause negative outcomes
- The key is creating NUANCED flaws: professional-sounding but fundamentally wrong

**BE CREATIVE - Don't just soften the language, RETHINK the misconception:**
- If original was too extreme ("ignore all feedback"), don't just write "minimize feedback"
- Instead, CREATE A NEW ANGLE: "Gather feedback through surveys, then prioritize operational data in final decisions"
- SHIFT the flaw TYPE: If timing was wrong before, try wrong stakeholder focus or incomplete approach now
- USE PROFESSIONAL NUANCE: Add constraints, timeframes, or rationales that sound reasonable

1. DO NOT change the questionText or the correct answer (isCorrect: true)

2. FIX THESE SPECIFIC ISSUES (read the feedback above carefully):

   IF PLAUSIBILITY FAILED → Make distractors MORE realistic and nuanced:
   - Read the feedback to identify which distractors scored low on plausibility
   - REMOVE extreme/absolute language: NO "solely", "only", "never", "always", "ignoring", "without any", "completely", "entirely"
   - AVOID obviously flawed approaches that anyone would reject (e.g., "ignore stakeholders", "skip compliance")
   - CREATE nuanced misconceptions that sound professionally reasonable but miss key elements
   - Example BAD: "Ignore team concerns to focus on operations" (too extreme)
   - Example GOOD: "Prioritize operational metrics while scheduling team discussions for next quarter" (sounds professional but delays engagement)
   - Think: Would a mid-level professional consider this approach? If no, it's too extreme

   IF MISCONCEPTION FAILED → Ensure DISTINCT error categories WITHIN {expected_incorrect} DISTRACTORS:
   - Each distractor must represent a DIFFERENT type of error:
     %%MISCONCEPTION_CATEGORIES_SHORT_BLOCK%%
   - If feedback says "missing METHOD/APPROACH", REPLACE your weakest distractor with one representing METHOD/APPROACH
   - If feedback says "missing COMPLETENESS", REPLACE your weakest distractor with one representing COMPLETENESS
   - DO NOT ADD a 5th, 6th, or 7th distractor - you must keep exactly {expected_incorrect} distractors total

   IF DISTINCTIVENESS FAILED → Make options more different:
   - Read which options are too similar in the feedback
   - Rewrite them to address DIFFERENT aspects of the problem

   IF INCORRECTNESS FAILED → Make distractors unambiguously wrong for THIS context:
   - Read the feedback to see which distractors are "debatable" or "could be argued as partially correct"
   - For EACH distractor, ensure it VIOLATES the learning objective in a clear way
   - Add specific elements that create measurable negative outcomes
   - Example BAD: "Focus on efficiency" (too vague, could be argued as valid)
   - Example GOOD: "Focus exclusively on short-term efficiency metrics without evaluating long-term stakeholder impact" (clearly misses key requirements)
   - Test: Could a skilled professional defend this approach? If yes, make the flaw more explicit

   IF CLARITY FAILED → Fix vague/ambiguous wording:
   - Replace vague terms with specific, precise language
   - Remove contradictory phrases

   IF PARALLELISM FAILED → Match word counts and structure:
   - Count words in each option
   - Target similar word counts across all options (±10–15% of each other).
   - Ensure distractors have comparable clause depth and specificity.
   - Use similar grammatical structure (all start with verb, etc.)

   IF SCENARIO_COHERENCE FAILED → Align with scenario context:
   - Read which options contradict the scenario
   - Rewrite them to fit naturally within the scenario constraints

3. **PRESERVE what passed** - Don't change aspects that validators approved

4. **Return COMPLETE question** with ALL fields intact

**CRITICAL CONSTRAINT - ANSWER COUNT:**
- STRICTLY follow the Locale: {questions_locale}
- You MUST return EXACTLY {expected_total} answer options total
- EXACTLY {expected_correct} with "isCorrect": true (DO NOT CHANGE THE CORRECT ANSWER)
- EXACTLY {expected_incorrect} with "isCorrect": false (the distractors you're fixing)
- DO NOT ADD extra distractors beyond {expected_incorrect}
- If feedback says "add a distractor for X", REPLACE an existing weak distractor with one representing that misconception category, don't add a 5th/6th option

STRICTLY only Return JSON in the following format:
{{
  "questionText": "...",
  "answer": [
    {{"answerText": "...", "isCorrect": true/false, "explanation": "..."}}
  ]
}}

NO explanations, NO reasoning, NO additional fields, NO markdown.
""".replace("%%MISCONCEPTION_CATEGORIES_SHORT_BLOCK%%", MISCONCEPTION_CATEGORIES_SHORT_BLOCK.replace("\n", "\n     "))

STEM_CLARITY_BIAS_SYSTEM_PROMPT = """
You are an assessment quality evaluator.
Your task is to evaluate a generated question stem for two quality dimensions: Clarity and Bias/Fairness.

SCOPE:
- Evaluate the question stem ONLY. Do NOT evaluate answer options (they are not provided).
- Do NOT suggest corrections or rewrites in your output.

DIMENSION 1 — CLARITY:
Detect whether the stem is double-barrelled, confusing, ambiguous, or otherwise unclear.
- Double-barrelled: asks two or more distinct questions in one stem.
- Confusing: wording makes it hard to determine what is being asked.
- Ambiguous: multiple reasonable interpretations of what is being asked.

DIMENSION 2 — BIAS AND FAIRNESS:
Detect cultural, gender, socioeconomic, or language bias that could unfairly advantage or disadvantage certain learners.
- Cultural bias: assumes knowledge of specific cultures, traditions, or regional contexts not essential to the learning objective.
- Gender bias: uses gendered assumptions, stereotypes, or unnecessarily gendered language.
- Socioeconomic bias: assumes access to resources, experiences, or backgrounds not universal.
- Language bias: uses idioms, colloquialisms, or phrasing that may be inaccessible to non-native speakers without adding assessment value.

VERDICT RULES:
- Give a SINGLE overall verdict of PASS or FAIL for the stem.
- The verdict is FAIL if the stem fails EITHER clarity OR bias/fairness. It is PASS only if it passes BOTH.
- If FAIL, provide a reason of 1-2 sentences naming the dimension(s) and the specific issue.
- If PASS, set reason to an empty string.

Respond ONLY with valid JSON. No markdown, no preamble."""

STEM_CLARITY_BIAS_USER_PROMPT = """
INPUTS:
Question type: {question_type}
Question stem: {question_stem}

Return ONLY this JSON structure:
{{
  "verdict": "PASS" or "FAIL",
  "reason": "<1-2 sentences when FAIL naming the dimension and specific issue; empty string when PASS>"
}}
"""

VALIDATION_SYSTEM_MESSAGE = "You are an assessment quality evaluator focused on {validation_check_role}. Always respond with valid JSON: {{\"score\": 0-100, \"pass\": true/false, \"feedback\": \"string\"}}."

DISTRACTOR_BATCH_CORRECTION_USER_PROMPT = """
You are correcting distractors (incorrect answer options) that failed quality validation checks.

**YOUR GOAL**: Raise failed check scores to meet pass thresholds
- Read each score and threshold carefully below
- Focus most on checks with the largest gaps
- Each check must score >= its threshold to pass

ORIGINAL QUESTION:
{original_question}

VALIDATION RESULTS:
Strictly follow the Locale: {questions_locale}
FAILED CHECKS (MUST FIX):
{all_feedback}

PASSED CHECKS (PRESERVE THESE):
{passed_feedback}
"""