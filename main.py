import json
from pprint import pprint
import re

# Configuration
BUILTIN_IMPORTS = [
    "player",
    "game",
    "entity",
    "math",
    "control",
    "item"
]

# Token definitions
class TokenType:
    COMMENT = 'COMMENT'
    KEYWORD = 'KEYWORD'
    TYPE = 'TYPE'
    IDENTIFIER = 'IDENTIFIER'
    STRING = 'STRING'
    FLOAT = 'FLOAT'
    INT = 'INT'
    BOOL = 'BOOL'
    NULL = 'NULL'
    OP = 'OP'
    SYMBOL = 'SYMBOL'
    WHITESPACE = 'WHITESPACE'

class Token:
    def __init__(self, type, value, position=None):
        self.type = type
        self.value = value
        self.position = position  # (line, column)
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

# Tokenizer (Lexer)
class Lexer:
    def __init__(self, source_code):
        self.source = source_code
        self.tokens = []
        
    def tokenize(self):
        """Split the source into tokens."""
        # This manual character-by-character tokenization will handle no-space event<Join> correctly
        tokens = []
        i = 0
        line_num = 1
        col_num = 1
        
        keywords = ["import", "event", "func", "proc", "return", "if", "else", "while", "for", "in", "break", "continue"]
        types = ["int", "float", "bool", "str", "list", "dict"]
        bool_literals = ["true", "false"]
        
        while i < len(self.source):
            char = self.source[i]
            pos = (line_num, col_num)
            
            # Handle whitespace
            if char.isspace():
                if char == '\n':
                    line_num += 1
                    col_num = 1
                else:
                    col_num += 1
                i += 1
                continue
                
            # Handle comments
            if char == '/' and i + 1 < len(self.source) and self.source[i + 1] == '/':
                # Skip to end of line
                i += 2
                col_num += 2
                while i < len(self.source) and self.source[i] != '\n':
                    i += 1
                    col_num += 1
                continue
                
            # Handle identifiers and keywords
            if char.isalpha() or char == '_':
                start = i
                while i < len(self.source) and (self.source[i].isalnum() or self.source[i] == '_'):
                    i += 1
                    col_num += 1
                
                ident = self.source[start:i]
                if ident in keywords:
                    tokens.append(Token(TokenType.KEYWORD, ident, pos))
                elif ident in types:
                    tokens.append(Token(TokenType.TYPE, ident, pos))
                elif ident in bool_literals:
                    tokens.append(Token(TokenType.BOOL, ident, pos))
                elif ident == "null":
                    tokens.append(Token(TokenType.NULL, ident, pos))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, ident, pos))
                continue
                
            # Handle numbers
            if char.isdigit():
                start = i
                is_float = False
                while i < len(self.source) and (self.source[i].isdigit() or self.source[i] == '.'):
                    if self.source[i] == '.':
                        is_float = True
                    i += 1
                    col_num += 1
                
                num = self.source[start:i]
                if is_float:
                    tokens.append(Token(TokenType.FLOAT, num, pos))
                else:
                    tokens.append(Token(TokenType.INT, num, pos))
                continue
                
            # Handle strings
            if char == '"':
                start = i
                i += 1
                col_num += 1
                while i < len(self.source) and self.source[i] != '"':
                    if self.source[i] == '\\' and i + 1 < len(self.source):
                        i += 2  # Skip escaped character
                        col_num += 2
                    else:
                        i += 1
                        col_num += 1
                        
                if i >= len(self.source):
                    raise Exception(f"Unterminated string starting at {pos}")
                    
                i += 1  # Include closing quote
                col_num += 1
                tokens.append(Token(TokenType.STRING, self.source[start:i], pos))
                continue
                
            # Handle operators
            if char in "+-*/=<>!&|^":
                op = char
                # Check for multi-character operators
                if i + 1 < len(self.source):
                    next_char = self.source[i + 1]
                    if (char + next_char) in ["==", "!=", "<=", ">=", "&&", "||"]:
                        op = char + next_char
                        i += 2
                        col_num += 2
                    else:
                        i += 1
                        col_num += 1
                else:
                    i += 1
                    col_num += 1
                    
                tokens.append(Token(TokenType.OP, op, pos))
                continue
                
            # Handle symbols
            if char in "[]{}();,:.":
                tokens.append(Token(TokenType.SYMBOL, char, pos))
                i += 1
                col_num += 1
                continue
                
            # Handle unknown characters
            raise Exception(f"Unexpected character '{char}' at {pos}")
            
        return tokens

# AST Node types
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self):
        self.segments = {}
        
    def add_segment(self, name, segment):
        self.segments[name] = segment
        
    def __repr__(self):
        return f"Program({len(self.segments)} segments)"

class Segment(ASTNode):
    def __init__(self, name, kind, params=None):
        self.name = name
        self.kind = kind  # event, func, proc
        self.params = params or {}
        self.body = []
        
    def add_statement(self, stmt):
        self.body.append(stmt)
        
    def __repr__(self):
        return f"{self.kind}({self.name}, {len(self.body)} statements)"

class FunctionCall(ASTNode):
    def __init__(self, name, args=None, import_name=None):
        self.name = name
        self.args = args or []
        self.import_name = import_name
        
    def __repr__(self):
        import_str = f"{self.import_name}." if self.import_name else ""
        return f"Call({import_str}{self.name}, {len(self.args)} args)"

class Assignment(ASTNode):
    def __init__(self, variable, expression):
        self.variable = variable
        self.expression = expression
        
    def __repr__(self):
        return f"Assign({self.variable}, {self.expression})"

class Return(ASTNode):
    def __init__(self, expression=None):
        self.expression = expression
        
    def __repr__(self):
        return f"Return({self.expression})"

class BinaryOp(ASTNode):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right
        
    def __repr__(self):
        return f"BinOp({self.operator}, {self.left}, {self.right})"

class Literal(ASTNode):
    def __init__(self, value, type_name):
        self.value = value
        self.type = type_name
        
    def __repr__(self):
        return f"{self.type}({self.value})"
    
class ListLiteral(ASTNode):
    def __init__(self, elements):
        self.elements = elements
    def __repr__(self):
        return f"List({self.elements})"

class DictLiteral(ASTNode):
    def __init__(self, pairs):
        self.pairs = pairs  # list of (key, value) tuples
    def __repr__(self):
        return f"Dict({self.pairs})"


class Variable(ASTNode):
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return f"Var({self.name})"

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
        self.ast = Program()
        
    def parse(self):
        """Parse the token stream into an AST."""
        while not self.is_at_end():
            self.parse_top_level()
        return self.ast
            
    def parse_top_level(self):
        """Parse top-level constructs (events, functions, procedures)."""
        if self.match(TokenType.KEYWORD, ['event', 'func', 'proc']):
            segment_kind = self.previous().value
            
            # Fix: Changed from SYMBOL to OP for the angle brackets
            self.consume(TokenType.OP, '<', f"Expected '<' after segment type '{segment_kind}'")
            segment_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.OP, '>', "Expected '>' after segment name")
            
            # Parse parameters
            self.consume(TokenType.SYMBOL, '(', "Expected '(' for parameters")
            params = self.parse_parameters()
            self.consume(TokenType.SYMBOL, ')', "Expected ')' after parameters")
            
            # Create segment
            segment = Segment(segment_name, segment_kind, params)
            
            # Parse body
            self.consume(TokenType.SYMBOL, '{', "Expected '{' for segment body")
            
            while not self.check(TokenType.SYMBOL, '}') and not self.is_at_end():
                statement = self.parse_statement()
                if statement:
                    segment.add_statement(statement)
                    
            self.consume(TokenType.SYMBOL, '}', "Expected '}' to close segment body")
                    
            self.ast.add_segment(segment_name, segment)
        else:
            self.advance()  # Skip tokens outside of segments
    
    def parse_parameters(self):
        """Parse function parameters."""
        params = {}
        
        if self.check(TokenType.SYMBOL, ')'):
            return params
            
        while True:
            param_name = self.consume(TokenType.IDENTIFIER).value
            param_type = None
            
            if self.match(TokenType.TYPE):
                param_type = self.previous().value
                
            params[param_name] = param_type
            
            if not self.match(TokenType.SYMBOL, [',']):
                break
                
        return params
    
    def parse_statement(self):
        """Parse a statement."""
        if self.match(TokenType.KEYWORD, ['return']):
            return self.parse_return_statement()
        elif self.match(TokenType.IDENTIFIER):
            # Check if this is an assignment or a function call
            identifier = self.previous().value
            
            if self.match(TokenType.OP, ['=']):
                expression = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';', "Expected ';' after assignment")
                return Assignment(identifier, expression)
            else:
                # Put the identifier back and parse as expression
                self.current -= 1
                expr = self.parse_expression()
                self.consume(TokenType.SYMBOL, ';', "Expected ';' after expression statement")
                return expr
        else:
            if not self.is_at_end():
                self.advance()  # Skip unrecognized statements
            return None
            
    def parse_return_statement(self):
        """Parse a return statement."""
        if self.check(TokenType.SYMBOL, ';'):
            self.advance()  # Consume ;
            return Return(None)
            
        expr = self.parse_expression()
        self.consume(TokenType.SYMBOL, ';', "Expected ';' after return expression")
        return Return(expr)
            
    def parse_expression(self):
        """Parse an expression."""
        return self.parse_additive()
        
    def parse_additive(self):
        """Parse addition/subtraction expressions."""
        expr = self.parse_multiplicative()
        
        while self.match(TokenType.OP, ['+', '-']):
            operator = self.previous().value
            right = self.parse_multiplicative()
            expr = BinaryOp(operator, expr, right)
            
        return expr
        
    def parse_multiplicative(self):
        """Parse multiplication/division expressions."""
        expr = self.parse_primary()
        
        while self.match(TokenType.OP, ['*', '/']):
            operator = self.previous().value
            right = self.parse_primary()
            expr = BinaryOp(operator, expr, right)
            
        return expr
        
    def parse_primary(self):
        """Parse primary expressions (literals, variables, function calls)."""
        if self.match(TokenType.INT):
            return Literal(int(self.previous().value), 'int')
            
        if self.match(TokenType.FLOAT):
            return Literal(float(self.previous().value), 'float')
            
        if self.match(TokenType.STRING):
            return Literal(self.previous().value.strip('"'), 'string')
            
        if self.match(TokenType.BOOL):
            return Literal(self.previous().value == 'true', 'bool')
            
        if self.match(TokenType.NULL):
            return Literal(None, 'null')
            
        if self.match(TokenType.IDENTIFIER):
            name = self.previous().value
            
            # Check if this is a function call
            if self.check(TokenType.SYMBOL, '('):
                return self.finish_function_call(name)
                
            # Check if it's a method call (obj.method())
            if self.check(TokenType.SYMBOL, '.'):
                return self.parse_method_call(name)
                
            # Otherwise it's a variable
            return Variable(name)
        
        # List literal
        if self.match(TokenType.SYMBOL, ['[']):
            elems = []
            if not self.check(TokenType.SYMBOL, ']'):
                while True:
                    elems.append(self.parse_expression())
                    if not self.match(TokenType.SYMBOL, [',']):
                        break
            self.consume(TokenType.SYMBOL, ']', "Expected ']' after list elements")
            return ListLiteral(elems)

        # Dict literal
        if self.match(TokenType.SYMBOL, ['{']):
            pairs = []
            if not self.check(TokenType.SYMBOL, '}'):
                while True:
                    # keys must be identifiers or strings
                    if self.match(TokenType.STRING):
                        key = self.previous().value.strip('"')
                    elif self.match(TokenType.IDENTIFIER):
                        key = self.previous().value
                    else:
                        raise Exception("Expected identifier or string as dict key")
                    self.consume(TokenType.SYMBOL, ':', "Expected ':' after dict key")
                    value = self.parse_expression()
                    pairs.append((key, value))
                    if not self.match(TokenType.SYMBOL, [',']):
                        break
            self.consume(TokenType.SYMBOL, '}', "Expected '}' after dict literal")
            return DictLiteral(pairs)
        if self.match(TokenType.SYMBOL, ['(']):
            expr = self.parse_expression()
            self.consume(TokenType.SYMBOL, ')', "Expected ')' after expression")
            return expr
            
        if self.is_at_end():
            raise Exception("Unexpected end of input while parsing expression")
        else:
            raise Exception(f"Unexpected token {self.peek().type} '{self.peek().value}' at {self.peek().position}")
    
    def parse_method_call(self, object_name):
        """Parse method calls (obj.method())."""
        self.consume(TokenType.SYMBOL, '.', "Expected '.' for method call")
        method_name = self.consume(TokenType.IDENTIFIER).value
        
        # Determine if this is a library/import method or an object method
        return self.finish_function_call(method_name, object_name)
    
    def finish_function_call(self, name, import_name=None):
        """Parse function call arguments."""
        self.consume(TokenType.SYMBOL, '(', "Expected '('")
        args = []
        
        if not self.check(TokenType.SYMBOL, ')'):
            while True:
                args.append(self.parse_expression())
                
                if not self.match(TokenType.SYMBOL, [',']):
                    break
                    
        self.consume(TokenType.SYMBOL, ')', "Expected ')' after arguments")
        return FunctionCall(name, args, import_name)
        
    # Helper methods for token management
    def match(self, type, values=None):
        """Check if current token matches type and value."""
        if self.is_at_end():
            return False
            
        if self.peek().type != type:
            return False
            
        if values is not None and self.peek().value not in values:
            return False
            
        self.advance()
        return True
        
    def check(self, type, value=None):
        """Check if current token is of given type and optionally value."""
        if self.is_at_end():
            return False
        
        if self.peek().type != type:
            return False
            
        if value is not None:
            if isinstance(value, list):
                return self.peek().value in value
            else:
                return self.peek().value == value
                
        return True
        
    def consume(self, type, value=None, error_message=None):
        """Consume token of expected type/value or throw error."""
        if self.check(type, value):
            return self.advance()
            
        if error_message is None:
            error_message = f"Expected {type}" + (f" '{value}'" if value else "")
            
        if self.is_at_end():
            raise Exception(f"{error_message} but reached end of file")
        else:
            token = self.peek()
            raise Exception(f"{error_message} but found {token.type} '{token.value}' at {token.position}")
        
    def advance(self):
        """Advance to next token."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
        
    def is_at_end(self):
        """Check if we've reached the end of the token stream."""
        return self.current >= len(self.tokens)
        
    def peek(self):
        """Look at current token without consuming it."""
        return self.tokens[self.current]
        
    def previous(self):
        """Get the last consumed token."""
        return self.tokens[self.current - 1]

class ASTSerializer:
    """Convert AST to serializable dictionary format."""
    
    @classmethod
    def serialize(cls, node):
        """Convert an AST node to a serializable dictionary."""
        if isinstance(node, Program):
            return {name: cls.serialize_segment(segment) for name, segment in node.segments.items()}
        return cls.serialize_node(node)
    
    @classmethod
    def serialize_segment(cls, segment):
        """Serialize a segment node."""
        return {
            "params": segment.params,
            "body": [cls.serialize_node(stmt) for stmt in segment.body]
        }
    
    @classmethod
    def serialize_node(cls, node):
        """Serialize a specific AST node."""
        if isinstance(node, FunctionCall):
            # call_data = {
            #     "action": "call",
            #     "data": {
            #         "module": node.import_name or "code",
            #         "name": node.name,
            #         "args": [cls.serialize_node(arg) for arg in node.args]
            #     }
            # }
            call_data = [(node.import_name or "code")+"."+ node.name] + [cls.serialize_node(arg) for arg in node.args]
            return call_data
            
        elif isinstance(node, Assignment):
            return {
                "action": "assign",
                "data": {
                    "var": node.variable,
                	"expr": cls.serialize_node(node.expression)
                }
            }
            
        elif isinstance(node, Return):
            return {"return": cls.serialize_node(node.expression) if node.expression else None}
            
        elif isinstance(node, BinaryOp):
            return [node.operator, cls.serialize_node(node.left), cls.serialize_node(node.right)]
            
        elif isinstance(node, Literal):
            return node.value
            
        elif isinstance(node, Variable):
            return node.name
        elif isinstance(node, ListLiteral):
            return [cls.serialize_node(e) for e in node.elements]
        elif isinstance(node, DictLiteral):
            return {k: cls.serialize_node(v) for k,v in node.pairs}

                    
        return str(node)

def parse_file(filename):
    """Parse a file containing the custom language."""
    with open(filename, 'r') as f:
        code = f.read()
    return parse_code(code)

def parse_code(code):
    """Parse a string containing the custom language."""
    # Lexical analysis
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    
    # Debug print tokens for troubleshooting
    # print("Tokens after lexical analysis:")
    # for idx, token in enumerate(tokens[:20]):  # Print first 20 tokens
    #     print(f"{idx}: {token.type} '{token.value}' at {token.position}")
    
    # Parsing
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Serialize for output
    return ASTSerializer.serialize(ast)

if __name__ == "__main__":
    # Test with file
    try:
        code = open("main.dfs", 'r').read()
        print(f"Parsing file with content:\n{code[:100]}...")  # Print first 100 chars to verify
        # result = json.dumps(parse_code(code), indent=4, sort_keys=True, default=lambda v: "null" if v is None else v)
        print(parse_code(code))
    except FileNotFoundError:
        print("File 'main.dfs' not found. Please provide a valid file.")
    except Exception as e:
        print(f"Error parsing: {e}")
        import traceback
        traceback.print_exc()