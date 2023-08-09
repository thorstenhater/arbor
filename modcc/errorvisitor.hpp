#pragma once

#include <iostream>
#include "visitor.hpp"
#include "expression.hpp"
#include <libmodcc/export.hpp>

class ARB_LIBMODCC_API ErrorVisitor : public Visitor {
public:
    ErrorVisitor(std::string const& m)
        : module_name_(m)
    {}

    using Visitor::visit;

    void visit(Expression *e)            override;
    void visit(ProcedureExpression *e)   override;
    void visit(FunctionExpression *e)    override;
    void visit(UnaryExpression *e)       override;
    void visit(BinaryExpression *e)      override;
    void visit(CallExpression *e)        override;
    void visit(ReactionExpression *e)    override;
    void visit(StoichExpression *e)      override;
    void visit(StoichTermExpression *e)  override;
    void visit(CompartmentExpression *e) override;
    void visit(PDiffExpression *e)       override;

    void visit(BlockExpression *e)       override;
    void visit(InitialBlock *e)          override;
    void visit(IfExpression *e)          override;

    int num_errors()   {return num_errors_;}
    int num_warnings() {return num_warnings_;}
private:
    template <typename ExpressionType>
    void print_error(ExpressionType *e) {
        if(e->has_error()) {
            auto header = red("error: ")
                        + white(pprintf("% % ", module_name_, e->location()));
            std::cout << header << "\n  "
                      << e->error_message()
                      << std::endl;
            num_errors_++;
        }
        if(e->has_warning()) {
            auto header = purple("warning: ")
                        + white(pprintf("% % ", module_name_, e->location()));
            std::cout << header << "\n  "
                      << e->warning_message()
                      << std::endl;
            num_warnings_++;
        }
    }

    std::string module_name_;
    int num_errors_ = 0;
    int num_warnings_ = 0;
};

