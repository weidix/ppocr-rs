//! CPU execution of backend-independent image preprocessing plans.

mod kernels;

#[cfg(any(feature = "cpu", test))]
use crate::ocr::DetectorInputPlan;
use crate::{
    RgbImage,
    ocr::{Point, RecognitionInputPlan},
};
use rayon::prelude::*;

use kernels::{Kernel, Normalization, RowPlan};

#[cfg(any(feature = "cpu", test))]
const DETECTOR_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
#[cfg(any(feature = "cpu", test))]
const DETECTOR_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];
const RECOGNIZER_MEAN_BGR: [f32; 3] = [0.5, 0.5, 0.5];
const RECOGNIZER_STD_BGR: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Clone, Debug)]
pub(crate) struct PreparedInput {
    pub(crate) data: Vec<f32>,
    pub(crate) batch: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl PreparedInput {
    pub(crate) fn shape(&self) -> [usize; 4] {
        [self.batch, 3, self.height, self.width]
    }
}

#[cfg(any(feature = "cpu", test))]
pub(crate) fn prepare_detector(image: &RgbImage, plan: DetectorInputPlan) -> PreparedInput {
    normalized_bgr(
        image,
        plan.corners(),
        plan.input_width(),
        plan.input_height(),
        plan.input_width(),
        &DETECTOR_MEAN_BGR,
        &DETECTOR_STD_BGR,
    )
}

pub(crate) fn prepare_recognizer(
    image: &RgbImage,
    plans: &[RecognitionInputPlan],
) -> PreparedInput {
    let width = plans.first().map_or(0, |plan| plan.input_width());
    let mut data = Vec::new();
    for plan in plans {
        debug_assert_eq!(plan.input_width(), width);
        data.extend(
            normalized_bgr(
                image,
                plan.corners(),
                plan.input_width(),
                plan.input_height(),
                plan.content_width(),
                &RECOGNIZER_MEAN_BGR,
                &RECOGNIZER_STD_BGR,
            )
            .data,
        );
    }
    PreparedInput {
        data,
        batch: plans.len(),
        height: crate::ocr::RECOGNIZER_INPUT_HEIGHT,
        width,
    }
}

fn normalized_bgr(
    image: &RgbImage,
    corners: [Point; 4],
    canvas_width: usize,
    canvas_height: usize,
    content_width: usize,
    mean: &[f32; 3],
    standard_deviation: &[f32; 3],
) -> PreparedInput {
    let plane_len = canvas_height * canvas_width;
    let mut data = vec![0.0; 3 * plane_len];
    let (blue, green_red) = data.split_at_mut(plane_len);
    let (green, red) = green_red.split_at_mut(plane_len);
    let kernel = Kernel::detect();
    let normalization = Normalization::new(*mean, *standard_deviation);
    let corners = corners.map(|point| [point.0, point.1]);

    blue.par_chunks_mut(canvas_width)
        .zip(green.par_chunks_mut(canvas_width))
        .zip(red.par_chunks_mut(canvas_width))
        .enumerate()
        .for_each(|(y, ((blue, green), red))| {
            kernel.preprocess_row(
                image.pixels(),
                RowPlan {
                    source_width: image.width() as usize,
                    source_height: image.height() as usize,
                    corners,
                    destination_y: y,
                    destination_height: canvas_height,
                    content_width,
                    normalization,
                },
                blue,
                green,
                red,
            );
        });

    PreparedInput {
        data,
        batch: 1,
        height: canvas_height,
        width: canvas_width,
    }
}

#[cfg(test)]
fn bilinear_quad(polygon: [Point; 4], u: f32, v: f32) -> Point {
    let top = add(scale(polygon[0], 1.0 - u), scale(polygon[1], u));
    let bottom = add(scale(polygon[3], 1.0 - u), scale(polygon[2], u));
    add(scale(top, 1.0 - v), scale(bottom, v))
}

#[cfg(test)]
fn add(left: Point, right: Point) -> Point {
    Point(left.0 + right.0, left.1 + right.1)
}

#[cfg(test)]
fn scale(point: Point, factor: f32) -> Point {
    Point(point.0 * factor, point.1 * factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ocr::recognition_input_plan,
        pixels::{from_fn, solid},
    };

    #[test]
    fn detector_uses_32_pixel_aligned_inputs() {
        let image = solid(1_920, 1_080, [0, 0, 0]);
        let plan = DetectorInputPlan::new(&image, Some(736)).expect("detector plan");
        let prepared = prepare_detector(&image, plan);
        assert_eq!(prepared.shape(), [1, 3, 416, 736]);
        assert!((plan.transform().map_x_to_source(736.0) - 1_920.0).abs() < 1e-4);
        assert!((plan.transform().map_y_to_source(416.0) - 1_080.0).abs() < 1e-4);
    }

    #[test]
    fn recognizer_pads_and_tracks_content_width() {
        let image = solid(80, 20, [127, 127, 127]);
        let plan = recognition_input_plan(
            [
                Point(0.0, 0.0),
                Point(80.0, 0.0),
                Point(80.0, 20.0),
                Point(0.0, 20.0),
            ],
            320,
        )
        .expect("recognizer plan");
        let prepared = prepare_recognizer(&image, &[plan]);
        assert_eq!(prepared.shape(), [1, 3, 48, 320]);
        assert_eq!(plan.content_width(), 192);
        let expected = (127.0 / 255.0 - 0.5) / 0.5;
        assert!((prepared.data[0] - expected).abs() < 1e-6);
        assert_eq!(prepared.data[plan.content_width()], 0.0);
    }

    #[test]
    fn fused_parallel_preprocess_matches_the_scalar_pipeline() {
        let image = from_fn(37, 29, |x, y| {
            [
                (x * 7 + y * 3) as u8,
                (x * 2 + y * 5) as u8,
                (x * 3 + y * 7) as u8,
            ]
        });
        let corners = [
            Point(2.25, 1.75),
            Point(34.5, 3.0),
            Point(32.75, 27.0),
            Point(1.0, 25.5),
        ];
        let width = 41;
        let height = 23;
        let prepared = normalized_bgr(
            &image,
            corners,
            width,
            height,
            width,
            &DETECTOR_MEAN_BGR,
            &DETECTOR_STD_BGR,
        );
        let plane_len = width * height;

        for y in 0..height {
            let v = (y as f32 + 0.5) / height as f32;
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let point = bilinear_quad(corners, u, v);
                let pixel = image.sample_bilinear(point.0 - 0.5, point.1 - 0.5);
                let bgr = [pixel[2], pixel[1], pixel[0]];
                for channel in 0..3 {
                    let expected = (f32::from(bgr[channel]) / 255.0 - DETECTOR_MEAN_BGR[channel])
                        / DETECTOR_STD_BGR[channel];
                    let actual = prepared.data[channel * plane_len + y * width + x];
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "channel {channel}, pixel ({x}, {y}): {actual} != {expected}"
                    );
                }
            }
        }
    }
}
