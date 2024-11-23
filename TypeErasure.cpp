// Type Erasure Shapes Example
// As per https://www.youtube.com/watch?v=jKt6A3wnDyI

#include <iostream>
#include <memory>
#include <vector>

#include <catch2/catch_all.hpp>

namespace type_erasure
{
    class Shape
    {
    private:
        struct ShapeConcept
        {
            ShapeConcept() = default;
            ShapeConcept(const ShapeConcept &) = default;
            ShapeConcept(ShapeConcept &&) = default;
            auto operator=(const ShapeConcept &s) -> ShapeConcept & = default;
            auto operator=(ShapeConcept &&s) -> ShapeConcept & = default;
            virtual ~ShapeConcept() = default;

            [[nodiscard]] virtual auto clone() const -> std::unique_ptr<ShapeConcept> = 0;
            virtual void draw() const = 0;
        };

        template <typename ConcreteShape, typename DrawStrategy>
        struct ShapeModel : public ShapeConcept
        {
            ShapeModel(ConcreteShape shape, DrawStrategy strategy)
                : m_shape{std::move(shape)}, m_drawStrategy{std::move(strategy)}
            {
            }

            [[nodiscard]] auto clone() const -> std::unique_ptr<ShapeConcept> override
            {
                return std::make_unique<ShapeModel>(*this);
            }

            void draw() const override
            {
                m_drawStrategy(m_shape);
            }

            ConcreteShape m_shape;
            DrawStrategy m_drawStrategy;
        };

        friend void draw(const Shape &shape)
        {
            shape.m_pImpl->draw();
        }

        std::unique_ptr<ShapeConcept> m_pImpl;

    public:
        template <typename ConcreteShape, typename DrawStrategy>
        Shape(ConcreteShape shape, DrawStrategy strategy)
            : m_pImpl{std::make_unique<ShapeModel<ConcreteShape, DrawStrategy>>(std::move(shape), std::move(strategy))}
        {
        }

        ~Shape() = default;

        // Special member functions
        Shape(const Shape &s)
            : m_pImpl{s.m_pImpl->clone()}
        {
        }

        auto operator=(const Shape &s) -> Shape &
        {
            m_pImpl = s.m_pImpl->clone();
            return *this;
        }

        Shape(Shape &&s) = default;
        auto operator=(Shape &&s) -> Shape & = default;
    };

    void drawAllShapes(const std::vector<Shape> &shapes)
    {
        for (const auto &shape : shapes)
        {
            draw(shape);
        }
    }

} // namespace type_erasure

// Concrete shapes
// ================
struct Square
{
    double side;
};

struct Rectangle
{
    double width;
    double height;
};

struct Circle
{
    double radius;
};

// Draw strategies
// ================

const auto drawSquare = [](const Square &square)
{
    std::cout << "Square with side " << square.side << '\n';
};

const auto drawRectangle = [](const Rectangle &rect)
{
    std::cout << "Rectangle with width " << rect.width
              << " and height " << rect.height << '\n';
};

const auto drawCircle = [](const Circle &circle)
{
    std::cout << "Circle with radius " << circle.radius << '\n';
};

// TEST CASES
//===========

TEST_CASE("Type Erasure")
{
    using namespace type_erasure;
    // Create some shapes
    std::vector<Shape> shapes;
    shapes.emplace_back(Circle{2.0}, drawCircle);
    shapes.emplace_back(Square{1.5}, drawSquare);
    shapes.emplace_back(Rectangle{4.2, 5.8}, drawRectangle);
    // Draw all shapes
    std::cout << '\n'
              << "SHAPES:" << '\n';
    drawAllShapes(shapes);
}
