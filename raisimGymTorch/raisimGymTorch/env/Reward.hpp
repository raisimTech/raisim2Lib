//
// Created by jemin on 20. 9. 22..
//

#ifndef _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
#define _RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_

#include <cstddef>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>
#include "Yaml.hpp"


namespace raisim {

struct RewardElement {
  float coefficient;
  float reward;
  float integral;
};

class Reward {
 public:
  Reward (std::initializer_list<std::string> names) {
    for(const auto& nm: names)
      addReward(nm, 1.f);
  }

  Reward () = default;

  void initializeFromConfigurationFile(const Yaml::Node& cfg) {
    rewards_.clear();
    rewardNames_.clear();
    rewardIndex_.clear();
    for(auto rw = cfg.Begin(); rw != cfg.End(); rw++) {
      rewardNames_.push_back((*rw).first);
      rewardIndex_[rewardNames_.back()] = rewards_.size();
      rewards_.push_back(raisim::RewardElement());
      RSFATAL_IF((*rw).second.IsNone() || (*rw).second["coeff"].IsNone(),
                 "Node " + (*rw).first + " or its coefficient doesn't exist");
      rewards_.back().coefficient = (*rw).second["coeff"].template As<float>();
      rewards_.back().reward = 0.f;
      rewards_.back().integral = 0.f;
    }
  }

  const float& operator [] (const std::string& name) {
    const auto it = rewardIndex_.find(name);
    RSFATAL_IF(it == rewardIndex_.end(), name<<" was not found in the configuration file")
    return rewards_[it->second].reward;
  }

  void record (const std::string& name, float reward, bool accumulate = false) {
    const auto it = rewardIndex_.find(name);
    RSFATAL_IF(it == rewardIndex_.end(), name<<" was not found in the configuration file")
    RSISNAN_MSG(reward, name<<" is nan")

    auto& entry = rewards_[it->second];
    if(!accumulate)
      entry.reward = 0.f;
    entry.reward += reward * entry.coefficient;
    entry.integral += entry.reward;
  }

  float sum() {
    float sum = 0.f;
    for(auto& rw: rewards_)
      sum += rw.reward;

    return sum;
  }

  void setZero() {
    for(auto& rw: rewards_)
      rw.reward = 0.f;
  }

  void reset() {
    for(auto& rw: rewards_) {
      rw.integral = 0.f;
      rw.reward = 0.f;
    }
  }

  int getSize() const { return static_cast<int>(rewards_.size()); }
  const std::vector<std::string>& getNames() const { return rewardNames_; }

  void fillRewardValues(float* out, int size) const {
    RSFATAL_IF(size != static_cast<int>(rewards_.size()), "Reward vector size mismatch")
    for (std::size_t i = 0; i < rewards_.size(); i++)
      out[i] = rewards_[i].reward;
  }

 private:
  void addReward(const std::string& name, float coefficient) {
    rewardNames_.push_back(name);
    rewardIndex_[name] = rewards_.size();
    rewards_.push_back(raisim::RewardElement{coefficient, 0.f, 0.f});
  }

  std::vector<raisim::RewardElement> rewards_;
  std::vector<std::string> rewardNames_;
  std::unordered_map<std::string, std::size_t> rewardIndex_;
};

}  // namespace raisim

#endif //_RAISIM_GYM_TORCH_RAISIMGYMTORCH_ENV_REWARD_HPP_
