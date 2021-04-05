#################################################
#                  Const                        #
#################################################

DIGITS = '0123456789'
STRINGCHAR = 'abcdefghijklmnopqrstuvwxyz'

#################################################
#                  Errors                       #
#################################################

class Error:
    def __init__(self, pos_start, pos_end ,error_name, details):
        self.error_name = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        result += f'  File {self.pos_start.fn}, Line {self.pos_start.ln + 1}'
        return result

class IllegalCharError(Error):
    def __init__(self,pos_start, pos_end, details):
        super().__init__(pos_start, pos_end,'Illegal Charecter', details)    

class InvalidSyntaxError(Error):
    def __init__(self,pos_start, pos_end, details):
        super().__init__(pos_start, pos_end,'Invalid syntax', details)  
        
class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Runtime Error', details)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

#################################################
#                  Token                        #
#################################################
TT_INT = 'TT_INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_LPRAN = 'LPRAN'
TT_RPRAN = 'RPRAN'
TT_EOF = 'EOF'

class Token:
    def __init__(self, type_, value = None, pos_start = None, pos_end = None):
        self.type_ = type_
        self.value = value
        
        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        
        if pos_end:
            self.pos_end = pos_end


    def __repr__(self):
        if self.value: return f'{self.type_}:{self.value}'
        return f'{self.type_}'

#################################################
#                  Pos                          #
#################################################

class possition:
    def __init__(self, idx, ln, cl,fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.cl = cl
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, curent_char=None):
        self.idx += 1
        self.cl += 1
        
        if curent_char == '\n':
            self.ln += 1
            self.cl = 0

        return self
    def copy(self):
        return possition(self.idx,self.ln,self.cl,self.fn,self.ftxt)


#################################################
#                  Lexer                        #
#################################################


class Lexer:
    def __init__(self, fn ,text):
        self.text = text
        self.fn = fn
        self.pos = possition(-1,0,-1,fn,text)
        self.curent_char = None
        self.Advance()

    def Advance(self):
        self.pos.advance(self.curent_char)
        self.curent_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_token(self):
        tokens = []
        
        while self.curent_char != None:
            if self.curent_char in ' \t':
                self.Advance()
            elif self.curent_char in DIGITS:
                tokens.append(self.make_number())
                self.Advance()
            elif self.curent_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos,))
                self.Advance()
            elif self.curent_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.Advance()
            elif self.curent_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.Advance()
            elif self.curent_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.Advance()
            elif self.curent_char == '(':
                tokens.append(Token(TT_LPRAN, pos_start=self.pos))
                self.Advance()
            elif self.curent_char == ')':
                tokens.append(Token(TT_RPRAN, pos_start=self.pos))
                self.Advance()
            elif self.curent_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.Advance()
            else:
                # return a error
                pos_start = self.pos.copy()
                char = self.curent_char
                self.Advance
                return [],  IllegalCharError(pos_start,self.pos,char)
        
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.curent_char != None and self.curent_char in DIGITS + '.':
            if self.curent_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
                self.Advance()
            else:
                num_str += self.curent_char
                self.Advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

#################################################
#                   Nodes                       #
#################################################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.rigth_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.rigth_node.pos_end
    
    def __repr__(self):
        return f'{self.left_node , self.op_tok , self.rigth_node}'
   
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'
    
     
#################################################
#                 Parser resul                  #
#################################################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None

	def register(self, res):
		if isinstance(res, ParseResult):
			if res.error: self.error = res.error
			return res.node

		return res

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		self.error = error
		return self


#################################################
#                   Parser                      #
#################################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_isx = -1
        self.advance()

    def advance(self):
        self.tok_isx += 1
        if self.tok_isx < len(self.tokens):
            self.curent_tok = self.tokens[self.tok_isx]
        return self.curent_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.curent_tok.type_ != TT_EOF:
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start,self.curent_tok.pos_end, "Expected '+', '-', '*' or '/'"))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.curent_tok

        if tok.type_ in (TT_INT, TT_FLOAT):
           res.register(self.advance())
           return res.success(NumberNode(tok))
           return res.failure(InvalidSyntaxError(tok.pos_start,tok.pos_end,"Expected int or float"))
        
        elif tok.type_ == TT_LPRAN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.curent_tok.type_ == TT_RPRAN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(tok.pos_start,tok.pos_end,"Expected ')'"))
        
            return res.failure(InvalidSyntaxError(self.curent_tok.pos_start,self.curent_tok.pos_end, "Expected '+', '-'or '('"))



    def power(self):
        return self.bin_op(self.atom, (TT_POW, ), self.factor)


    def factor(self):
        res = ParseResult()
        tok = self.curent_tok

        if tok.type_ in (TT_PLUS,TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok,factor))
        return self.power()
        

    def term(self):
        return self.bin_op(self.factor, (TT_MUL,TT_DIV))


    def expr(self):
         return self.bin_op(self.term,(TT_PLUS,TT_MINUS))
    

    def bin_op(self, func_a, ops, func_b=None):
       if func_b == None:
           func_b = func_a
       res = ParseResult()
       left = res.register(func_a())
       if res.error: return res

       while self.curent_tok.type_ in ops:
            op_tok = self.curent_tok
            res.register(self.advance())
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left,op_tok,right)
       return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
	def __init__(self):
		self.value = None
		self.error = None

	def register(self, res):
		if res.error: self.error = res.error
		return res.value

	def success(self, value):
		self.value = value
		return self

	def failure(self, error):
		self.error = error
		return self

#################################################
#                   Number                      #
#################################################

class Number:
	def __init__(self, value):
		self.value = value
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		if isinstance(other, Number):
			return Number(self.value + other.value).set_context(self.context), None

	def subbed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value - other.value).set_context(self.context), None

	def multed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value * other.value).set_context(self.context), None

	def dived_by(self, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(
					other.pos_start, other.pos_end,
					'Division by zero',
					self.context
				)

			return Number(self.value / other.value).set_context(self.context), None

	def powed_by(self, other):
		if isinstance(other, Number):
			return Number(self.value ** other.value).set_context(self.context), None

	def __repr__(self):
		return str(self.value)
   

#######################################
# CONTEXT
#######################################

class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos

#################################################
#                   Interpreter                 #
#################################################

class Interpreter:
	def Visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	

	def visit_NumberNode(self, node, context):
		return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))


	def visit_BinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.Visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.Visit(node.rigth_node, context))
		if res.error: return res



		if node.op_tok.type_ == TT_PLUS:
			result, error = left.added_to(right)
		elif node.op_tok.type_ == TT_MINUS:
			result, error = left.subbed_by(right)
		elif node.op_tok.type_ == TT_MUL:
			result, error = left.multed_by(right)
		elif node.op_tok.type_ == TT_DIV:
			result, error = left.dived_by(right)
		elif node.op_tok.type_ == TT_POW:
			result, error = left.powed_by(right)
        


		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.error: return res

		error = None

		if node.op_tok.type == TT_MINUS:
			number, error = number.multed_by(Number(-1))

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))


#################################################
#                   Run                         #
#################################################

def Run(fn,text):
    # Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_token()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error

	# Run program
	interpreter = Interpreter()
	context = Context('<program>')
	result = interpreter.Visit(ast.node, context)

	return result.value, result.error
