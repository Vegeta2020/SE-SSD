#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>

using namespace std;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const float EPS = 1e-8;


struct Point {
    float x, y;
    Point() {}
    Point(double _x, double _y){
        x = _x, y = _y;
    }

    void set(float _x, float _y){
        x = _x; y = _y;
    }

    Point operator +(const Point &b)const{
        return Point(x + b.x, y + b.y);
    }

    Point operator -(const Point &b)const{
        return Point(x - b.x, y - b.y);
    }
};

inline int point_cmp(const Point &a, const Point &b, const Point &center){
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

inline float cross(const Point &a, const Point &b){
    return a.x * b.y - a.y * b.x;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0){
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2){
    int ret = std::min(p1.x,p2.x) <= std::max(q1.x,q2.x)  &&
              std::min(q1.x,q2.x) <= std::max(p1.x,p2.x) &&
              std::min(p1.y,p2.y) <= std::max(q1.y,q2.y) &&
              std::min(q1.y,q2.y) <= std::max(p1.y,p2.y);
    return ret;
}

inline int check_in_box2d(const float *box, const Point &p){
    //params: box (5) [x1, y1, x2, y2, angle]
    const float MARGIN = 1e-5;

    float center_x = (box[0] + box[2]) / 2;
    float center_y = (box[1] + box[3]) / 2;
    float angle_cos = cos(-box[4]), angle_sin = sin(-box[4]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
    float rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
#ifdef DEBUG
    printf("box: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", box[0], box[1], box[2], box[3], box[4]);
    printf("center: (%.3f, %.3f), cossin(%.3f, %.3f), src(%.3f, %.3f), rot(%.3f, %.3f)\n", center_x, center_y,
            angle_cos, angle_sin, p.x, p.y, rot_x, rot_y);
#endif
    return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN && rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

inline int check_in_box2d_input3d(const float *box, const Point &p){
    //params: box (7) [x1, y1, z1, x2, y2, z2, angle]
    const float MARGIN = 1e-5;

    float center_x = (box[0] + box[3]) / 2;
    float center_y = (box[1] + box[4]) / 2;
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
    float rot_y = -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
#ifdef DEBUG
    printf("box: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", box[0], box[1], box[3], box[4], box[6]);
    printf("center: (%.3f, %.3f), cossin(%.3f, %.3f), src(%.3f, %.3f), rot(%.3f, %.3f)\n", center_x, center_y,
            angle_cos, angle_sin, p.x, p.y, rot_x, rot_y);
#endif
    return (rot_x > box[0] - MARGIN && rot_x < box[3] + MARGIN && rot_y > box[1] - MARGIN && rot_y < box[4] + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans){
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    }
    else{
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p){
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
    float new_y = -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

inline float box_overlap(const float *box_a, const float *box_b, const int input_2d=1){
    // params: box_a (5) [x1, y1, x2, y2, angle] or box_a [x1, y1, z1, x2, y2, z2, angle]
    // params: box_b (5) [x1, y1, x2, y2, angle] or box_b [x1, y1, z1, x2, y2, z2, angle]
    // params: imply data format of box_a/b

    float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3], a_angle = box_a[4];
    float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3], b_angle = box_b[4];

    if(input_2d==0){
        a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[3], a_y2 = box_a[4], a_angle = box_a[6];
        b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[3], b_y2 = box_b[4], b_angle = box_b[6];
    }

    Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
    Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);
#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
           b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (input_2d==1 && check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (input_2d==1 && check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
        if (input_2d==0 && check_in_box2d_input3d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (input_2d==0 && check_in_box2d_input3d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}


inline float iou_bev(const float *box_a, const float *box_b){
    // params: box_a (5) [x1, y1, x2, y2, angle]
    // params: box_b (5) [x1, y1, x2, y2, angle]
    float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

int boxes_overlap_bev_cpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap){
    // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
    // params boxes_b: (M, 5)
    // params ans_overlap: (N, M)

    CHECK_CONTIGUOUS(boxes_a);
    CHECK_CONTIGUOUS(boxes_b);
    CHECK_CONTIGUOUS(ans_overlap);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_overlap_data = ans_overlap.data<float>();

    for(int i=0; i < num_a; i++){
        for(int j=0; j < num_b; j++){
            ans_overlap_data[i*num_b + j] = box_overlap(boxes_a_data + 5*i, boxes_b_data + 5*j);
        }
    }
    return 1;
}


int boxes_iou_bev_cpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
    // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
    // params boxes_b: (M, 5)
    // params ans_iou: (N, M)

    CHECK_CONTIGUOUS(boxes_a);
    CHECK_CONTIGUOUS(boxes_b);
    CHECK_CONTIGUOUS(ans_iou);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_iou_data = ans_iou.data<float>();
    for(int i=0; i < num_a; i++){
        for(int j=0; j < num_b; j++){
            ans_iou_data[i*num_b + j] = iou_bev(boxes_a_data + 5*i, boxes_b_data + 5*j);
        }
    }
    return 1;
}

int boxes_iou3d_cpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
    // params: box_a/box_b (N/M, 7)  [x1, y1, z1, x2, y2, z2, angle]
    // params ans_overlap: (N, M)

    CHECK_CONTIGUOUS(boxes_a);
    CHECK_CONTIGUOUS(boxes_b);
    CHECK_CONTIGUOUS(ans_iou);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * box_a = boxes_a.data<float>();
    const float * box_b = boxes_b.data<float>();
    float * ans_iou_data = ans_iou.data<float>();
    float va, vb, min_height, max_height, delta_height, v_overlap;
    for(int i=0; i < num_a; i++){
        for(int j=0; j < num_b; j++){
            va = (box_a[i*7+3] - box_a[i*7+0]) * (box_a[i*7+4] - box_a[i*7+1]) * (box_a[i*7+5] - box_a[i*7+2]);
            vb = (box_b[j*7+3] - box_b[j*7+0]) * (box_b[j*7+4] - box_b[j*7+1]) * (box_b[j*7+5] - box_b[j*7+2]);
            min_height = fmaxf(box_a[i*7+2], box_b[j*7+2]);
            max_height = fminf(box_a[i*7+5], box_b[j*7+5]);
            delta_height = fmaxf(max_height - min_height, EPS);
            if(delta_height == EPS){
                ans_iou_data[i*num_a + j] = 0.0;
            }
            v_overlap = box_overlap(box_a+7*i, box_b+7*j, 0) * delta_height;
            ans_iou_data[i*num_b + j] = v_overlap / fmaxf(va + vb - v_overlap, EPS);
        }
    }
    return 1;
}
