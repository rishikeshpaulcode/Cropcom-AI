# store generated suggestion into training corpus if quality criteria satisfied
training_corpus = []
training_corpus.append({
    "system": "You are an experienced argultural assistant having emence domain knowledge. You are given the real time farm (crop and soil) and weather data. You task is to suggest the farmer a set of instructions they must follow today.",
    "user": "Realtime farm data:\n" + " | ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in curr_farm_data.items()),
    "assistant": teacher_response.summary + "\n\n" + teacher_response.instructions
})

# save generated suggestions to jsonl file
with open(output_file, "w", encoding="utf-8") as f:
    for entry in training_corpus:
        formatted_entry = {
            "messages": [
                {"role": "system", "content": entry["system"]},
                {"role": "user", "content": entry["user"]},
                {"role": "assistant", "content": entry["assistant"]}
            ]
        }
        # write each JSON object as a single line
        f.write(json.dumps(formatted_entry, ensure_ascii=False) + "\n")
