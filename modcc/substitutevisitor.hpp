#pragma once

#include "visitor.hpp"

class ReplaceVariablesVisitor: public Visitor {
    public:
        ReplaceVariablesVisitor(std::string const& from, Symbol* to) : from_(from), to_(to) {}
        ~ReplaceVariablesVisitor() {}

        void visit(Expression *e) override {};

        void visit(ProcedureExpression *e) override {
            for(auto& expression: e->args()) {
                expression->accept(this);
            }
        }

        void visit(FunctionExpression *e) override {
            for(auto& expression: e->args()) {
                expression->accept(this);
            }
        }

        void visit(UnaryExpression *e) override {
            e->expression()->accept(this);
        }

        void visit(BinaryExpression *e) override {
            e->lhs()->accept(this);
            e->rhs()->accept(this);
        }

        void visit(AssignmentExpression *e) override {
            auto symbol = e->lhs()->is_identifier()->symbol();
            if (symbol && !e->lhs()->is_derivative()) {
                if (symbol->name() == from_) {
                    e->lhs()->is_identifier()->symbol(to_);
                }
            }
            e->rhs()->accept(this);
        }

        void visit(DerivativeExpression* e) override {}


        void visit(IdentifierExpression* e) override {
            auto symbol = e->is_identifier()->symbol();
            if(symbol) {
                if (symbol->name() == from_) {
                    e->symbol(to_);
                } 
            }
        }

    private:
        std::string from_;
        Symbol* to_;
};
