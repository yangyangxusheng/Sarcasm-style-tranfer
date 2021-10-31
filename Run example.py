from run_pplm import run_pplm_example

run_pplm_example(
    cond_text="the 10 carpet design styles you need",
    # when use fine-tuned gpt-2 starting words : sarcastic: xxxx xxx x xxx  or not_sarcastic xxx xxx xxx
    # when use gpt-2 starting words: xxx xxx xxx
    num_samples=1, # The number of generated sentences
    discrim='sarcasm', # Attribute
    discrim_weights='headlines_sarcasm_discriminator(7).pt', # The discriminator
    discrim_meta='sarcasm_classifier_head_meta.json', # config
    class_label='1', # '0' : unsarcastic, '1': sarcastic
    length=10, # The length of generated sentence
    stepsize=0.06, # Increase to intensify topic control, and decrease its value to soften the control.
    sample=True,
    num_iterations=10, # Cumulative interference
    gamma=1, # Parameters of the normalized gradient
    gm_scale=0.95, # Fuse the modified model and original model, default = 0.95, decrease gm_scale ->  Reduce repetition
    kl_scale=0.03, # Ensure language fluency, default = 0.01,  increase kl_scale  ->  Reduce repetition
    verbosity='regular' # The level of detail of the displayed information: quiet 0 , regular 1 , verbose 2 , very_verbose 3
)
"""
如果改变max_length , 未干扰的生成不变，只是截取前面的
而扰动过的句子会随着变短成更简洁的语义
step size 变小会减少控制， 变大会出现重复
增加迭代次数，也会早成重复
If you change max_length, the undisturbed generation remains unchanged, just intercept the previous
The disturbed sentences will be shortened into more concise semantics.
Step size becomes smaller, it reduces control, and becomes more repetitive.
Increase the number of iterations, it will be repeated early.
"""



