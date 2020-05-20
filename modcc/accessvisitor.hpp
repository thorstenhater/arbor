#pragma once

#include <unordered_set>

#include "visitor.hpp"

class AccessVisitor: public Visitor {
    public:
        const std::unordered_set<Symbol*>& reads()  const { return rds_; }
        const std::unordered_set<Symbol*>& writes() const { return wrs_; }

        virtual void visit(Expression *e) override {}

        virtual void visit(DerivativeExpression *e) override {}

        virtual void visit(ProcedureExpression *e) override {
            for(auto& expression: e->args()) {
                expression->accept(this);
            }
        }

        virtual void visit(FunctionExpression *e) override {
            for(auto& expression: e->args()) {
                expression->accept(this);
            }
        }

        virtual void visit(BinaryExpression *e) override {
            e->lhs()->accept(this);
            e->rhs()->accept(this);
        }

        virtual void visit(UnaryExpression *e) override {
            e->expression()->accept(this);
        }

        virtual void visit(AssignmentExpression *e) override {
            auto symbol = e->lhs()->is_identifier()->symbol();
            if(!symbol) {
                throw compiler_exception(" undefined symbol", e->location());
            }
            if (!e->lhs()->is_derivative()) {
                switch (symbol->kind()) {
                    case symbolKind::variable:
                        wrs_.insert(symbol);
                        break;
                    case symbolKind::indexed_variable:
                        wrs_.insert(symbol);
                        break;
                    default:
                        break;
                }
            }
            e->rhs()->accept(this);
        }

        void visit(IdentifierExpression* e) override {
            auto symbol = e->symbol();
            if(!symbol) {
                throw compiler_exception(" undefined symbol", e->location());
            }
            switch (symbol->kind()) {
                case symbolKind::variable:
                    rds_.insert(symbol);
                    break;
                case symbolKind::indexed_variable:
                    rds_.insert(symbol);
                    break;
                default:
                    break;
            }
        }

    private:
        std::unordered_set<Symbol*> rds_;
        std::unordered_set<Symbol*> wrs_;
};
