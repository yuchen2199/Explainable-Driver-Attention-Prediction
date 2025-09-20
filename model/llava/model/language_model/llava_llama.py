#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPastAndFeature(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    loss_what: Optional[torch.FloatTensor] = None
    loss_why: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_features: Optional[List[torch.FloatTensor]] = None
    output_ids: torch.Tensor = None


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,             # (2,447)
        attention_mask: Optional[torch.Tensor] = None,  # (2,447)
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,      # (2,447)
        labels_what: Optional[torch.LongTensor] = None, # (2,447)
        labels_why: Optional[torch.LongTensor] = None,  # (2,447)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,    # True
        images: Optional[torch.FloatTensor] = None,     # (2,3,224,224)
        return_dict: Optional[bool] = None,
        # added
        clip_resize_list=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (   # False
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (    # True
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = ( # True
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            output_image_features,
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            labels_what,
            labels_why,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, labels_what, labels_why, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # Q + A
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (2,702,4096)
        logits = self.lm_head(hidden_states)    # (2,702,32004)
        output_ids = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)

        loss = None
        loss_what = None
        loss_why = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()     # (2,701,32004)
            shift_labels = labels[..., 1:].contiguous()         # (2,701)
            if labels_what is not None:
                shift_labels_what = labels_what[..., 1:].contiguous()  # (2,701)
            if labels_why is not None:
                shift_labels_why = labels_why[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)    # (1450,32004)
            shift_labels = shift_labels.view(-1)    # (1450)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            if labels_what is not None:
                shift_labels_what = shift_labels_what.view(-1)
                shift_labels_what = shift_labels_what.to(shift_logits.device)
            if labels_why is not None:
                shift_labels_why = shift_labels_why.view(-1)
                shift_labels_why = shift_labels_why.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)

            if labels_what is not None:
                loss_what = loss_fct(shift_logits, shift_labels_what)
            if labels_why is not None:
                loss_why = loss_fct(shift_logits, shift_labels_why)
                # if np.isnan(loss_why.cpu().detach().numpy()):
                #     print('loss_why is nan')
                # if torch.isnan(loss_why).any():
                #     print('loss_why is nan')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPastAndFeature(
            loss=loss,
            loss_what=loss_what,
            loss_why=loss_why,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,
            attentions=outputs.attentions,
            image_features=output_image_features,
            output_ids=output_ids
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
