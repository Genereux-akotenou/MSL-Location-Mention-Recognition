def formating_prompt(rule_set: List[str], input_str: str, label_str: str, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    rule_str = "\n".join(rule_set)

    # Message 1
    usr_msg1 = "You are given a user utterance that may contain Location Mention and you have to perform Location Mention Recognition" \
        "task (LMR). Given a microblogging posts during emergencies, the task is to recognize all location mentions. " \
        "You are also given a list of entity types representing Location type we want to recognise. " \
        "Your task is to detect and identify all instances of the supplied Location entity types in the user utterance. " \
        "The output must have the same content as the input. Only the tokens that match the Location entities in the " \
        "list should be enclosed within XML tags. The XML tag comes from the Location entities described in the list below. " \
        "For example, a city should be enclosed within <CITY></CITY> tags." \
        "Ensure that all entities are identified. Do not perform false identifications." \
        f"""\n\nList Of Entities\n{rule_str}"""\
        "\n\n" \
        "Are the instructions clear to you?"
    asst_msg1 = "Yes, the instructions are clear. I will identify and enclose within the corresponding XML tags, " \
        "all instances of the specified Location entity types in the user utterance. For example, " \
        "<CITY><Name of city></CITY>, <ROAD><A street, avenue, or highway that connects different locations.></ROAD>, etc. " \
        "leaving the rest of the user utterance unchanged."
    
    # Message 2
    usr_msg2 = "Florida Bahamas flooding shuts down train lines."
    asst_msg2 = "<STATE>Florida</STATE> <ISLAND>Bahamas</ISLAND> flooding shuts down train lines."
    
    # Message 3
    usr_msg3 = "Give a brief explanation of why your answer is correct."
    asst_msg3 = "I identified and enclosed within corresponding XML tags, all instances of the specified Location " \
        "entity types in the user utterance - The \"Florida\" state in U.S within the <STATE></STATE> tag, and " \
        "the Florida island \"Bahamas\" within the <ISLAND></ISLAND> tag. The rest of the user " \
        "utterance was left unchanged as it did not contain any other identified Location entities."
    
    # Message 4
    usr_msg4 = "Great! I am now going to give you another user utterance. Please detect Location entities in it " \
        "according to the previous instructions. Do not include an explanation in your answer."
    asst_msg4 = "Sure! Please give me the user utterance."

    # Encode in conversation
    messages = [
        {"role": "user", "content": usr_msg1},
        {"role": "assistant", "content": asst_msg1},
        {"role": "user", "content": usr_msg2},
        {"role": "assistant", "content": asst_msg2},
        {"role": "user", "content": usr_msg3},
        {"role": "assistant", "content": asst_msg3},
        {"role": "user", "content": usr_msg4},
        {"role": "assistant", "content": asst_msg4},
        {"role": "user", "content": input_str},
        {"role": "assistant", "content": label_str},
    ]

    return messages