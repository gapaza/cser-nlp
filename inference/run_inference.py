import config
import model.vocabulary as vocabulary
from model import get_requirement_decoder as get_model
import tensorflow as tf


num_samples = 5
sample_type = 'soft'  # 'hard' or 'soft'

def run():

    model = get_model(checkpoint_path=config.model_path)

    # --- Autoregressive Inference ---
    generated_requirements = []
    for idx in range(num_samples):
        output_tokens = autoregressive_inference(model, stype=sample_type)
        print('Generated Requirement '+str(idx)+':', ' '.join(output_tokens))


def autoregressive_inference(model, stype='hard'):
    input_tokens = [vocabulary.start_token_id]
    output_tokens = []
    for _ in range(vocabulary.max_seq_len):

        # 1. Create input tensor
        input_tensor = tf.expand_dims(input_tokens, axis=0)

        # 2. Pass through decoder
        inf_idx = len(input_tokens) - 1
        if stype == 'hard':
            next_token_id = hard_sample(model, input_tensor, inf_idx)
        else:
            next_token_id = soft_sample(model, input_tensor, inf_idx)

        # 3. Append to the current sequence
        input_tokens.append(next_token_id)

        # 4. Exit loop if "end-of-sequence" token is generated
        if next_token_id == vocabulary.end_token_id:
            break

        # 5. Append output to the current sequence
        next_token = vocabulary.id2token[next_token_id]
        output_tokens.append(next_token)

    return output_tokens


def hard_sample(model, input_tensor, inf_idx):
    decoder_output = model(input_tensor, training=False)
    last_token_logits = decoder_output[0, inf_idx, :]
    next_token_id = tf.argmax(last_token_logits).numpy()
    return next_token_id

def soft_sample(model, input_tensor, inf_idx):
    decoder_output = model(input_tensor, training=False)
    last_token_probs = decoder_output[:, inf_idx, :]
    last_token_log_probs = tf.math.log(last_token_probs + 1e-10)
    samples = tf.random.categorical(last_token_log_probs, 1)  # shape (batch, 1)
    next_token_id = tf.squeeze(samples, axis=-1)  # shape (batch,)
    return next_token_id.numpy()[0]






if __name__ == "__main__":
    run()



