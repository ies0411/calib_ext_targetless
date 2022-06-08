#ifndef __OPT_H__
#define __OPT_H__
#include <ceres/ceres.h>

#include <eigen3/Eigen/Core>

#include "common.h"
class vpnp_calib {
   public:
    vpnp_calib(VPnPData &p, Eigen::Matrix3d &inner, Eigen::Vector4d &distor) {
        pd = p;
        this->inner = inner;
        this->distor = distor;
    }
    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const {
        Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
        Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
        Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
        T uo = p_2[0] / p_2[2];
        T vo = p_2[1] / p_2[2];
        const T &fx = innerT.coeffRef(0, 0);
        const T &cx = innerT.coeffRef(0, 2);
        const T &fy = innerT.coeffRef(1, 1);
        const T &cy = innerT.coeffRef(1, 2);
        T xo = (uo - cx) / fx;
        T yo = (vo - cy) / fy;
        T r2 = xo * xo + yo * yo;
        T r4 = r2 * r2;
        T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
        T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
               distorT[3] * (r2 + xo * xo + xo * xo);
        T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
               distorT[2] * (r2 + yo * yo + yo * yo);
        T ud = fx * xd + cx;
        T vd = fy * yd + cy;
        if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
            residuals[0] = ud - T(pd.u);
            residuals[1] = vd - T(pd.v);
        } else {
            residuals[0] = ud - T(pd.u);
            residuals[1] = vd - T(pd.v);
            Eigen::Matrix<T, 2, 2> I =
                Eigen::Matrix<float, 2, 2>::Identity().cast<T>();
            Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();
            Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();
            Eigen::Matrix<T, 2, 2> V = n * nt;
            V = I - V;
            Eigen::Matrix<T, 2, 1> R = Eigen::Matrix<float, 2, 1>::Zero().cast<T>();
            R.coeffRef(0, 0) = residuals[0];
            R.coeffRef(1, 0) = residuals[1];
            R = V * R;
            residuals[0] = R.coeffRef(0, 0);
            residuals[1] = R.coeffRef(1, 0);
        }
        return true;
    }
    static ceres::CostFunction *Create(VPnPData &p, Eigen::Matrix3d &inner, Eigen::Vector4d &distor) {
        return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(new vpnp_calib(p, inner, distor)));
    }

   private:
    VPnPData pd;
    Eigen::Matrix3d inner;
    Eigen::Vector4d distor;
};
#endif