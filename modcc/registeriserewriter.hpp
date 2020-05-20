#pragma once

#include <vector>

#include "astmanip.hpp"
#include "visitor.hpp"
#include "accessvisitor.hpp"
#include "substitutevisitor.hpp"

class RegisteriseRewriter: public Visitor {
    public:
        virtual void visit(Expression *e) override {};
        virtual void visit(BinaryExpression *e) override {};
        virtual void visit(UnaryExpression *e) override {};

        virtual void visit(APIMethod* e) override { visit((ProcedureExpression*) e); }

        virtual void visit(ProcedureExpression* e) override {
            auto body = e->body();
            auto access = AccessVisitor();
            for (auto& s: body->statements()) {
                s->accept(&access);
            }

            std::cout << "Reading variables\n";
            for (auto& v: access.reads()) std::cout << " - " << v->name() << '\n';
            std::cout << "Writing variables\n";
            for (auto& v: access.writes()) std::cout << " - " << v->name() << '\n';

            std::vector<expression_ptr> locals;
            std::vector<expression_ptr> prologue;
            std::vector<expression_ptr> epilogue;

            auto sym_to_id = [](Symbol* sym) -> expression_ptr {
                auto id = make_expression<IdentifierExpression>(sym->location(), sym->name());
                id->is_identifier()->symbol(sym);
                return id;
            };

            for (auto& var: access.reads()) {
                if (access.writes().find(var) == access.writes().end()) { // Read only
                    auto assign = make_unique_local_assign(e->scope(), sym_to_id(var).get(), "reg_");
                    auto substitute = ReplaceVariablesVisitor{var->name(), assign.id->is_identifier()->symbol()};
                    for (auto& s: body->is_block()->statements()) {
                        s->accept(&substitute);
                    }
                    locals.push_back(std::move(assign.local_decl));
                    prologue.push_back(std::move(assign.assignment));
                }
            }

            for (auto& var: access.writes()) {
                if (access.reads().find(var) == access.reads().end()) { // Write only
                    auto decl = make_unique_local_decl(e->scope(), var->location(), "reg_");
                    std::cerr << "WO " << var->name() << " ->" << decl.id->is_identifier()->symbol()->name() << '\n';
                    auto substitute = ReplaceVariablesVisitor{var->name(), decl.id->is_identifier()->symbol()};
                    for (auto& s: body->is_block()->statements()) {
                        s->accept(&substitute);
                    }
                    locals.push_back(std::move(decl.local_decl));
                    epilogue.push_back(make_expression<AssignmentExpression>(var->location(),
                                                                             sym_to_id(var),
                                                                             std::move(decl.id)));
                } else { // Read and write
                    auto assign = make_unique_local_assign(e->scope(), sym_to_id(var).get(), "reg_");
                    auto substitute = ReplaceVariablesVisitor{var->name(), assign.id->is_identifier()->symbol()};
                    for (auto& s: body->is_block()->statements()) {
                        s->accept(&substitute);
                    }
                    locals.push_back(std::move(assign.local_decl));
                    prologue.push_back(std::move(assign.assignment));
                    epilogue.push_back(make_expression<AssignmentExpression>(var->location(),
                                                                             sym_to_id(var),
                                                                             std::move(assign.id)));
                }
            }

            // Dump in the newly created statements
            auto& stmnts = body->is_block()->statements();
            stmnts.insert(stmnts.end(),
                          std::make_move_iterator(epilogue.begin()),
                          std::make_move_iterator(epilogue.end()));

            stmnts.insert(stmnts.begin(),
                          std::make_move_iterator(prologue.begin()),
                          std::make_move_iterator(prologue.end()));

            stmnts.insert(stmnts.begin(),
                          std::make_move_iterator(locals.begin()),
                          std::make_move_iterator(locals.end()));
            // e->body(std::move(body));
        }
};
