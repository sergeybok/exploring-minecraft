import tensorflow as tf
import time

sess = tf.Session()
curr_episode_total_reward = 0

curr_episode_total_reward_placeholder = tf.placeholder(tf.float32, name='curr_episode_total_reward')


writer_op = tf.summary.FileWriter('./example_tf_graphs', sess.graph)
# curr_episode_total_reward_summary = tf.Summary()
# for i in range(50):
#     curr_episode_total_reward = (i+1)*10
#     curr_episode_total_reward_summary.value.add(tag='curr_episode_total_reward_summary', simple_value=curr_episode_total_reward)
#     writer_op.add_summary(curr_episode_total_reward_summary, i)

curr_episode_reward_summary = tf.summary.scalar("curr_episode_total_reward", curr_episode_total_reward_placeholder)
# summary_op = tf.summary.merge_all()

for i in range(50):
    curr_episode_total_reward = (i+1)*10
    # summary_val = sess.run([summary_op], feed_dict = {curr_episode_total_reward_placeholder:curr_episode_total_reward})
    time.sleep(10)
    summary_val, = sess.run([curr_episode_reward_summary], feed_dict = {curr_episode_total_reward_placeholder:curr_episode_total_reward})
    print(curr_episode_total_reward)
    writer_op.add_summary(summary_val, i+1)

writer_op.close()
sess.close()