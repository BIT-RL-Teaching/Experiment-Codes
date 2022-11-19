def PPO_update_actor_critic(self, s, a, logprob_a, adv, value_past, td_target, actor_critic, optimizer, optim_iter_num):
    '''
    注释仅注明与前文针对mujoco的PPO类不同的部分
    s：状态
    a：动作
    logprob_a：动作的对数概率
    adv：优势函数
    value_past：使用本批经验更新之前的状态价值估计
    td_target：critic网络待逼近的return， 有adv = td_target - value_past
    '''
    for j in range(optim_iter_num):
        index = slice(j * self.optim_batch_size, min((j + 1) * self.optim_batch_size, s.shape[0]))

        ## 过网络
        # 由于actor和critic共享卷积层，因此过网络会同时输出当前状态价值估计value_now
        # 和动作概率分布action_prob，前者将用于计算critic loss，后者将用于计算actor loss
        value_now, action_prob = actor_critic(s[index])
        ## 计算actor_loss
        distribution = Categorical(action_prob)
        logprob_a_now = distribution.log_prob(a[index].squeeze(-1)).unsqueeze(-1)  # shape = (batch, )
        dist_entropy = distribution.entropy()
        ratio = torch.exp(logprob_a_now - logprob_a[index])
        surr1 = ratio * adv[index]
        surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
        surr_loss = -torch.min(surr1, surr2)  # 取负号的原因是需要提升能获得较高优势函数的动作的概率
        actor_loss = torch.mean(surr_loss)

        ## 计算critic_loss
        # clip的作用在于确保更新后的价值估计不要离trust region更远
        # 详见 https://github.com/openai/baselines/issues/91
        value_now_clipped = value_past[index] + (value_now - value_past[index]).clamp(-self.clip_rate, self.clip_rate)
        value_loss_clipped = (value_now_clipped - td_target[index]).pow(2)
        value_loss = (value_now - td_target[index]).pow(2)
        critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()  # 这个0.5相当于critic loss coef再减半

        ## 反向传播
        # 代价函数由actor loss、critic loss、entropy项共同组成
        # entropy项取负号的原因是需要提升Actor网络输出的概率分布的熵，以鼓励探索
        optimizer.zero_grad()
        total_loss = critic_loss * self.critic_loss_coef \
                     + actor_loss \
                     - torch.mean(dist_entropy) * self.entropy_coef

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        optimizer.step()
