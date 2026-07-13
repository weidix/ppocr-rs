use candle_core::{Error, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LSTM, LSTMConfig, Linear, RNN, VarBuilder, rnn};

pub const CONV_KERNEL: usize = 3;
pub const CONV_STRIDE: usize = 2;
pub const CONV_PADDING: usize = 1;
pub const CONV_LAYERS: usize = 4;

pub struct Crnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    lstm_fw: LSTM,
    lstm_bw: LSTM,
    fc: Linear,
    height_feat: usize,
}

impl Crnn {
    pub fn new(vb: VarBuilder, num_classes: usize, image_height: usize) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: CONV_STRIDE,
            padding: CONV_PADDING,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(1, 64, CONV_KERNEL, conv_cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(64, 128, CONV_KERNEL, conv_cfg, vb.pp("conv2"))?;
        let conv3 = candle_nn::conv2d(128, 256, CONV_KERNEL, conv_cfg, vb.pp("conv3"))?;
        let conv4 = candle_nn::conv2d(256, 256, CONV_KERNEL, conv_cfg, vb.pp("conv4"))?;

        let height_feat = conv_repeat_out_size(image_height, CONV_LAYERS);
        if height_feat == 0 {
            return Err(Error::msg(format!(
                "invalid conv output height for input {image_height}"
            )));
        }

        let input_size = 256 * height_feat;

        let lstm_cfg_fw = LSTMConfig {
            direction: rnn::Direction::Forward,
            ..Default::default()
        };
        let lstm_cfg_bw = LSTMConfig {
            direction: rnn::Direction::Backward,
            ..Default::default()
        };
        let lstm_fw = rnn::lstm(input_size, 128, lstm_cfg_fw, vb.pp("lstm_fw"))?;
        let lstm_bw = rnn::lstm(input_size, 128, lstm_cfg_bw, vb.pp("lstm_bw"))?;
        let fc = candle_nn::linear(256, num_classes, vb.pp("fc"))?;

        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
            lstm_fw,
            lstm_bw,
            fc,
            height_feat,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.apply(&self.conv1)?.relu()?;
        let x = x.apply(&self.conv2)?.relu()?;
        let x = x.apply(&self.conv3)?.relu()?;
        let x = x.apply(&self.conv4)?.relu()?;

        let (_b, _c, h, _w) = x.dims4()?;
        if h != self.height_feat {
            return Err(Error::msg(format!(
                "unexpected feature height {h}, expected {}",
                self.height_feat
            )));
        }

        let x = x.transpose(1, 3)?.transpose(2, 3)?;
        let seq = x.flatten(2, 3)?;

        let states_fw = self.lstm_fw.seq(&seq)?;
        let states_bw = self.lstm_bw.seq(&seq)?;
        let out_fw = self.lstm_fw.states_to_tensor(&states_fw)?;
        let out_bw = self.lstm_bw.states_to_tensor(&states_bw)?;

        let outputs = Tensor::stack(&[out_fw, out_bw], 3)?;
        let outputs = outputs.flatten(2, 3)?;
        let logits = outputs.apply(&self.fc)?;
        Ok(logits)
    }
}

pub fn time_steps_for_width(width: usize) -> usize {
    conv_repeat_out_size(width, CONV_LAYERS)
}

fn conv_repeat_out_size(size: usize, layers: usize) -> usize {
    let mut out = size;
    for _ in 0..layers {
        out = conv_out_size(out, CONV_KERNEL, CONV_STRIDE, CONV_PADDING, 1);
    }
    out
}

fn conv_out_size(
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}
