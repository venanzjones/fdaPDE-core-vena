// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __CONSTEXPR_MATRIX_H__
#define __CONSTEXPR_MATRIX_H__

#include <array>

namespace fdapde {
namespace cexpr {

  // better triangular matrix arithmetic? enable triangular assignment for Upper and Lower views (not UnitUpper/Lower)
  // visitors, rowwise, colwwise iterators, sum, max, min, ...
  
template <int Rows, int Cols, typename Derived> struct MatrixBase;

template <typename Derived>
struct Transpose : public MatrixBase<Derived::Rows, Derived::Cols, Transpose<Derived>> {
    using XprType = Transpose<Derived>;
    using Base = MatrixBase<Derived::Rows, Derived::Cols, XprType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr Transpose(const Derived& xpr) : xpr_(xpr) { }
    constexpr Scalar operator()(int i, int j) const { return xpr_(j, i); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          xpr_.cols() == 1 || xpr_.rows() == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return xpr_[i];
    }
    constexpr int rows() const { return xpr_.cols(); }
    constexpr int cols() const { return xpr_.rows(); }
   protected:
    typename internals::ref_select<const Derived>::type xpr_;
};

[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix
[[maybe_unused]] constexpr int UnitUpper = 2;   // lower triangular view of matrix with ones on the diagonal
[[maybe_unused]] constexpr int UnitLower = 3;   // upper triangular view of matrix with ones on the diagonal

template <typename Derived, int ViewMode>
struct TriangularView : public MatrixBase<Derived::Rows, Derived::Cols, TriangularView<Derived, ViewMode>> {
    using XprType = TriangularView<Derived, ViewMode>;
    using Base = MatrixBase<Derived::Rows, Derived::Cols, XprType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr TriangularView() = default;
    constexpr TriangularView(const Derived& xpr) : xpr_(xpr) { }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }

    constexpr Scalar operator()(int i, int j) const {
        if constexpr (ViewMode == Upper) return i > j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == UnitUpper) return i > j ? 0 : (i == j ? 1 : xpr_(i, j));
        if constexpr (ViewMode == UnitLower) return i < j ? 0 : (i == j ? 1 : xpr_(i, j));
    }
   private:
    typename internals::ref_select<const Derived>::type xpr_;
};

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixBinOp : public MatrixBase<Lhs::Rows, Lhs::Cols, MatrixBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      std::is_same<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>::value,
      OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    fdapde_static_assert(Lhs::Rows == Rhs::Rows && Lhs::Cols == Rhs::Cols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    using XprType = MatrixBinOp<Lhs, Rhs, BinaryOperation>;
    using Base = MatrixBase<Lhs::Rows, Lhs::Cols, XprType>;
    using Scalar = typename Lhs::Scalar;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Lhs::Cols;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr MatrixBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) : lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(lhs_(i, j), rhs_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (Lhs::Cols == 1 && Rhs::Cols == 1) || (Lhs::Rows == 1 && Rhs::Rows == 1),
          THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
   protected:
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
    BinaryOperation op_;
};
template <typename Op1, typename Op2>
constexpr MatrixBinOp<Op1, Op2, std::plus<>>
operator+(const MatrixBase<Op1::Rows, Op1::Cols, Op1>& op1, const MatrixBase<Op2::Rows, Op2::Cols, Op2>& op2) {
    return MatrixBinOp<Op1, Op2, std::plus<>> {op1.derived(), op2.derived(), std::plus<>()};
}
template <typename Op1, typename Op2>
constexpr MatrixBinOp<Op1, Op2, std::minus<>>
operator-(const MatrixBase<Op1::Rows, Op1::Cols, Op1>& op1, const MatrixBase<Op2::Rows, Op2::Cols, Op2>& op2) {
    return MatrixBinOp<Op1, Op2, std::minus<>> {op1.derived(), op2.derived(), std::minus<>()};
}

template <typename Lhs, typename Rhs>
struct MatrixProduct : public MatrixBase<Lhs::Rows, Rhs::Cols, MatrixProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      std::is_same<typename Lhs::Scalar FDAPDE_COMMA typename Rhs::Scalar>::value,
      OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    fdapde_static_assert(Lhs::Cols == Rhs::Rows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
    using XprType = MatrixProduct<Lhs, Rhs>;
    using Base = MatrixBase<Lhs::Rows, Rhs::Cols, XprType>;
    using Scalar = typename Lhs::Scalar;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Rhs::Cols;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr MatrixProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) { }
    constexpr Scalar operator()(int i, int j) const {
        Scalar tmp {};
        for (int k = 0; k < Lhs::Cols; ++k) tmp += lhs_(i, k) * rhs_(k, j);
        return tmp;
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return rhs_.cols(); }
   protected:
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
};
template <typename Op1, typename Op2>
constexpr MatrixProduct<Op1, Op2>
operator*(const MatrixBase<Op1::Rows, Op1::Cols, Op1>& op1, const MatrixBase<Op2::Rows, Op2::Cols, Op2>& op2) {
    return MatrixProduct<Op1, Op2> {op1.derived(), op2.derived()};
}

template <int BlockRows_, int BlockCols_, typename Derived>
class MatrixBlock : public MatrixBase<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, Derived>> {
    fdapde_static_assert(
      BlockRows_ > 0 && BlockCols_ > 0 && BlockRows_ <= Derived::Rows && BlockCols_ <= Derived::Cols,
      INVALID_BLOCK_SIZES);
   public:
    using XprType = MatrixBlock<BlockRows_, BlockCols_, Derived>;
    using Base = MatrixBase<BlockRows_, BlockCols_, XprType>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr MatrixBlock(Derived& xpr, int start_row, int start_col) :
        start_row_(start_row), start_col_(start_col), xpr_(xpr) { }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr Scalar operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // block assignment
    template <int Rows_, int Cols_, typename RhsXprType>
    constexpr XprType& operator=(const MatrixBase<Rows_, Cols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows_ == BlockRows_ && Cols_ == BlockCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_BLOCK_ASSIGNMENT);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs(i, j); }
        }
        return *this;
    }
   protected:
    int start_row_ = 0, start_col_ = 0;
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename Scalar_, int Rows_, int Cols_>
class Matrix : public MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_>> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, EMPTY_MATRIX_IS_ILL_FORMED);
   public:
    using XprType = Matrix<Scalar_, Rows_, Cols_>;
    using Base = MatrixBase<Rows_, Cols_, XprType>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;   // whether to store this node by reference or by copy in an expression

    constexpr Matrix() : data_() {};
    constexpr explicit Matrix(const std::array<Scalar, Rows * Cols>& data) : data_(data) { }
    template <typename Derived>
    constexpr explicit Matrix(const MatrixBase<Rows_, Cols_, Derived>& derived) : data_() {
        fdapde_static_assert(
          std::is_same<Scalar FDAPDE_COMMA typename Derived::Scalar>::value, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols, YOU_ARE_ASSIGNING_MATRICES_OF_DIFFERENT_SIZES);
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { data_[i * Cols + j] = derived(i, j); }
        }
    }
    template <typename Callable>
    constexpr explicit Matrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA Rows * Cols>>,
          OPERANDS_HAVE_NON_CONVERTIBLE_SCALAR_TYPES);
        data_ = callable();
    }
    constexpr explicit Matrix(const Scalar_ (&data)[Rows * Cols]) : data_() {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { data_[i * Cols + j] = data[i * Cols + j]; }
        }
    }
    constexpr explicit Matrix(Scalar x) : data_() {   // 1D point constructor
        fdapde_static_assert(Base::size() == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_ONE_ELEMENT);
	data_[0] = x;
    }
    constexpr explicit Matrix(Scalar x, Scalar y) : data_() {   // 2D point constructor
        fdapde_static_assert(Base::size() == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_TWO_ELEMENTS);
	data_ = {x, y};
    }
    constexpr explicit Matrix(Scalar x, Scalar y, Scalar z) : data_() {   // 3D point constructor
        fdapde_static_assert(Base::size() == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_THREE_ELEMENTS);
	data_ = {x, y, z};
    }
    // static constructors
    static constexpr Matrix<Scalar, Rows, Cols> Constant(Scalar c) {
        Matrix<Scalar, Rows, Cols> m;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { m(i, j) = c; }
        }
        return m;
    }
    static constexpr Matrix<Scalar, Rows, Cols> Zero() { return Constant(Scalar(0)); }
    static constexpr Matrix<Scalar, Rows, Cols> Ones() { return Constant(Scalar(1)); }
    // const access
    constexpr Scalar operator()(int i, int j) const { return data_.at(i * Cols + j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_.at(i);
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return data_[i * Cols + j]; }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr const std::array<Scalar, Rows * Cols>& data() const { return data_; }
    // assignment operator
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr XprType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_MATRIX_ASSIGNMENT);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs(i, j); }
        }
        return *this;
    }
   private:
    std::array<Scalar, Rows * Cols> data_;
};

template <typename Derived>
struct Diagonal : public MatrixBase<Derived::Rows, Derived::Cols, Diagonal<Derived>> {
    using XprType = Diagonal<Derived>;
    using Base = MatrixBase<Derived::Rows, Derived::Cols, XprType>;
    static constexpr int MinSize = std::min(Derived::Rows, Derived::Cols);
    static constexpr int MaxSize = std::max(Derived::Rows, Derived::Cols);
    using Scalar = typename Derived::Scalar;

    constexpr Diagonal() = default;
    constexpr Diagonal(Derived& xpr) : xpr_(xpr) { }
    constexpr int rows() const { return Derived::Rows < Derived::Cols ? MinSize : MaxSize; }
    constexpr int cols() const { return Derived::Rows < Derived::Cols ? MaxSize : MinSize; }
    constexpr Scalar operator()(int i, int j) const { return i == j ? xpr_(i, i) : 0; }
    // assignment operator
    template <typename RhsType> constexpr XprType& operator=(const RhsType& rhs) {
        fdapde_static_assert(Base::size() == rhs.size(), INVALID_BLOCK_ASSIGNMENT);
        for (int i = 0; i < MinSize; ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
   private:
    typename internals::ref_select<Derived>::type xpr_;
};

template <int Rows, int Cols, typename Derived> struct MatrixBase {
    constexpr int size() const { return derived().rows() * derived().cols(); }
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // access operator on base type XprType
    constexpr auto operator()(int i, int j) const { return derived().operator()(i, j); }
    constexpr auto operator[](int i) const { return derived().operator[](i); }
    // send matrix to out stream (this is not constexpr evaluable)
    friend std::ostream& operator<<(std::ostream& out, const MatrixBase& m) {
        for (int i = 0; i < m.derived().rows() - 1; ++i) {
            for (int j = 0; j < m.derived().cols(); ++j) { out << m(i, j) << " "; }
            out << "\n";
        }
        // print last row without carriage return
        for (int j = 0; j < m.derived().cols(); ++j) { out << m(m.derived().rows() - 1, j) << " "; }
        return out;
    }
    // frobenius norm (L^2 norm of a matrix)
    constexpr double squared_norm() const {
        double norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += std::pow(derived().operator()(i, j), 2); }
        }
        return norm_;
    }
    constexpr double norm() const { return std::sqrt(squared_norm()); }
    // maximum norm (L^\infinity norm)
    constexpr double inf_norm() const {
        double norm_ = std::numeric_limits<double>::min();
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                double tmp = std::abs(derived().operator()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
    }
    // transpose
    constexpr Transpose<Derived> transpose() const { return Transpose<Derived>(derived()); }
    // block operations
    constexpr MatrixBlock<Rows, 1, Derived> col(int i) { return block<Rows, 1>(0, i); }
    constexpr MatrixBlock<Rows, 1, const Derived> col(int i) const {
        return MatrixBlock<Rows, 1, const Derived>(derived(), 0, i);
    }
    constexpr MatrixBlock<1, Cols, Derived> row(int i) { return block<1, Cols>(i, 0); }
    constexpr MatrixBlock<1, Cols, const Derived> row(int i) const {
        return MatrixBlock<1, Cols, const Derived>(derived(), i, 0);
    }
    template <int BlockRows, int BlockCols> constexpr MatrixBlock<BlockRows, BlockCols, Derived> block(int i, int j) {
        return MatrixBlock<BlockRows, BlockCols, Derived>(derived(), i, j);
    }
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> topRows(int i) {
        return block<BlockRows, Cols>(i, 0);
    }
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> bottomRows(int i) {
        return block<BlockRows, Cols>(Rows - i, 0);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> leftCols(int i) {
        return block<Rows, BlockCols>(0, i);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> rightCols(int i) {
        return block<Rows, BlockCols>(0, Cols - i);
    }
    // dot product
    template <int RhsRows, typename RhsDerived>
    constexpr auto dot(const MatrixBase<RhsRows, 1, RhsDerived>& rhs) const {
        fdapde_static_assert(Rows == 1 && Cols == RhsRows, INVALID_OPERANDS_DIMENSIONS_FOR_DOT_PRODUCT);
        fdapde_static_assert(
          std::is_same<typename RhsDerived::Scalar FDAPDE_COMMA typename Derived::Scalar>::value,
          OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
        typename Derived::Scalar dot_ = 0;
        for (int i = 0; i < Cols; ++i) { dot_ += derived().operator[](i) * rhs[i]; }
        return dot_;
    }
    // trace of matrix
    constexpr auto trace() const {
        fdapde_static_assert(Rows == Cols, CANNOT_COMPUTE_TRACE_OF_NON_SQUARE_MATRIX);
        typename Derived::Scalar trace_ = 0;
        for (int i = 0; i < Rows; ++i) trace_ += derived().operator()(i, i);
        return trace_;
    }
    // diagonal view of matrix expression
    constexpr Diagonal<const Derived> diagonal() const { return Diagonal<const Derived>(derived()); }
    constexpr Diagonal<Derived> diagonal() { return Diagonal<Derived>(derived()); }
    // triangular view of matrix expression
    template <int ViewMode> constexpr TriangularView<const Derived, ViewMode> triangular_view() const {
        return TriangularView<const Derived, ViewMode>(derived());
    }
    template <int ViewMode> constexpr TriangularView<Derived, ViewMode> triangular_view() {
        return TriangularView<Derived, ViewMode>(derived());
    }
   protected:
    // trait to detect if Xpr is a compile-time vector
    template <typename Xpr> struct is_vector {
        static constexpr bool value = (Xpr::Cols == 1);
    };
    template <typename Xpr> using is_vector_v = is_vector<Xpr>::value;
};

// comparison operators
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator==(const MatrixBase<Rows1, Cols1, XprType1>& op1, const MatrixBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) != op2.derived()(i, j)) return false;
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator!=(const MatrixBase<Rows1, Cols1, XprType1>& op1, const MatrixBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) == op2.derived()(i, j)) return false;
        }
    }
    return true;
}

template <int Size_> struct PermutationMatrix {
    using XprType = PermutationMatrix<Size_>;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression

    constexpr PermutationMatrix() = default;
    constexpr explicit PermutationMatrix(const std::array<int, Size_>& permutation) : permutation_(permutation) { }
    constexpr int rows() const { return Size_; }
    constexpr int cols() const { return Size_; }   // permutation matrix is square

    // left multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Size_; ++i) { permuted.row(i) = rhs.row(permutation_[i]); }
        return permuted;
    }
    // right multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr friend Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& lhs, const PermutationMatrix<Size_>& rhs) {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Size_; ++i) { permuted.col(i) = rhs.col(rhs.permutation()[i]); }
        return permuted;
    }
    constexpr const std::array<int, Size_>& permutation() const { return permutation_; }
   private:
    std::array<int, Size_> permutation_;
};

// alias export for constexpr-enabled vectors
template <typename Scalar_, int Rows_> using Vector = Matrix<Scalar_, Rows_, 1>;

template <typename Matrix, typename Rhs> constexpr auto backward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = rows - 1;
    res[i] = b[i] / A(i, i);
    i--;
    for (; i >= 0; --i) {
        Scalar tmp = 0;
        for (int j = i + 1; j < rows; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

template <typename Matrix, typename Rhs> constexpr auto forward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = 0;
    res[i] = b[i] / A(i, i);
    i++;
    for (; i < rows; ++i) {
        Scalar tmp = 0;
        for (int j = 0; j < i; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

// LU factorization of matrix with partial pivoting
template <typename MatrixType> class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    static constexpr int Size = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    MatrixType m_;
    PermutationMatrix<Size> P_;
   public:
    constexpr PartialPivLU() : m_(), P_() {};
    constexpr PartialPivLU(const MatrixType& m) : m_(m) { }

    // computes the LU factorization of matrix m with partial (row) pivoting
    template <typename XprType> constexpr void compute(const MatrixBase<Size, Size, XprType>& m) {
        m_ = m;
        std::array<int, Size> P;
        int pivot_index = 0;
        for (int i = 0; i < Size; ++i) {
            // find pivotal element
            Scalar pivot = 0;
            for (int j = i; j < Size; ++j) {
                if (pivot < std::abs(m_(j, i))) {
                    pivot = std::abs(m_(j, i));
                    pivot_index = j;
                }
            }
            // perform gaussian elimination step in place
            P[i] = pivot_index;
            for (int j = i; j < Size; ++j) {
                if (j != pivot_index) {
                    Scalar l = m_(j, i) / m_(pivot_index, i);
                    for (int k = i + 1; k < Size; ++k) { m_(j, k) = m_(j, k) - l * m_(pivot_index, k); }
                    m_(j, i) = l;
                }
            }
        }
        P_ = PermutationMatrix<Size>(P);
    }
    constexpr PermutationMatrix<Size> P() const { return P_; }
    // solve linear system Ax = b using A factorization PA = LU
    template <typename RhsType> constexpr Matrix<Scalar, Size, 1> solve(const RhsType& rhs) {
        fdapde_static_assert(
          rhs.rows() == Size && rhs.cols() == 1 && std::is_same_v<Scalar FDAPDE_COMMA typename RhsType::Scalar>,
          INVALID_RHS_VECTOR_OR_OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
        Matrix<Scalar, Size, 1> x;
        // evaluate U^{-1} * (L^{-1} * (P * rhs))
        x = P_ * rhs;
        x = forward_sub(m_.template triangular_view<UnitLower>(), x);
        x = backward_sub(m_.template triangular_view<Upper>(), x);
        return x;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // _CONSTEXPR_MATRIX_H__
